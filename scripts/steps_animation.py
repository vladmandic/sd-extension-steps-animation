# pylint: disable=import-error
import json
import os
import shutil
import string
import pathlib
import subprocess
import logging

import torch
import gradio as gr

from modules import scripts
from modules import shared
from modules.images import save_image
from modules.processing import create_infotext
from modules.sd_samplers import sample_to_image
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.sd_samplers_compvis import VanillaStableDiffusionSampler

from pkg_resources import resource_filename

# configurable section
video_rate = 30
name_length = 96
author = 'https://github.com/vladmandic'
cli_template = "{ffmpeg} -hide_banner -loglevel {loglevel} -hwaccel auto -y -framerate {framerate} -start_number {sequence} -i \"{inpath}/%7d-{short_name}.{extension}\" -r {videorate} {preset} {vfilters} {flags} -metadata title=\"{description}\" -metadata description=\"{info}\" -metadata author=\"stable-diffusion\" -metadata album_artist=\"{author}\" \"{outfile}\"" # note: <https://wiki.multimedia.cx/index.php/FFmpeg_Metadata>

presets = {
    'x264': '-vcodec libx264 -preset medium -crf 23 -pix_fmt yuv420p',
    'x265': '-vcodec libx265 -preset faster -crf 28 -pix_fmt yuv420p',
    'vpx-vp9': '-vcodec libvpx-vp9 -crf 34 -b:v 0 -deadline realtime -cpu-used 4',
    'aom-av1': '-vcodec libaom-av1 -crf 28 -b:v 0 -usage realtime -cpu-used 8 -pix_fmt yuv444p',
    'prores_ks': '-vcodec prores_ks -profile:v 3 -vendor apl0 -bits_per_mb 8000 -pix_fmt yuv422p10le',
}


# internal state variables
current_step = 0
current_preview_mode = 'undefined'
orig_callback_state = 'undefined'
temp_files = []
log = logging.getLogger('steps-animation')
log.setLevel(logging.INFO)


def safestring(text: str):
    lines = []
    for line in text.splitlines():
        lines.append(line.translate(str.maketrans('', '', string.punctuation)).strip())
    res = ', '.join(lines)
    return res[:1000]

def find_ffmpeg_binary():
    try:
        import google.colab
        return 'ffmpeg'
    except:
        pass
    for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
        try:
            package_path = resource_filename(package, 'binaries')
            files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0] if files else shutil.which('ffmpeg')
        except:
            return shutil.which('ffmpeg')

class Script(scripts.Script):
    def __init__(self): # pylint: disable=useless-super-delegation
        super().__init__()

    # script title to show in ui
    def title(self):
        return 'Steps animation'


    # is ui visible: process/postprocess triggers for always-visible scripts otherwise use run as entry point
    def show(self, is_img2img):
        return scripts.AlwaysVisible


    # ui components
    def ui(self, is_img2img): # pylint: disable=unused-argument
        with gr.Accordion('Steps animation', open = False, elem_id='steps-animation'):
            gr.HTML("""
                <a href="https://github.com/vladmandic/sd-extension-steps-animation/blob/main/README.md">
                Creates animation sequence from denoised intermediate steps with video frame interpolation to achieve desired animation duration</a><br>""")
            with gr.Row():
                is_enabled = gr.Checkbox(label = 'Script Enabled', value = False)
                codec = gr.Radio(label = 'Codec', choices = ['x264', 'x265', 'vpx-vp9', 'aom-av1', 'prores_ks'], value = 'x264')
                interpolation = gr.Radio(label = 'Interpolation', choices = ['none', 'mci', 'blend'], value = 'blend')
            with gr.Row():
                duration = gr.Slider(label = 'Duration', minimum = 0.5, maximum = 120, step = 0.1, value = 10)
                skip_steps = gr.Slider(label = 'Skip steps', minimum = 0, maximum = 100, step = 1, value = 0)
            with gr.Row():
                last_frame_duration = gr.Slider(label = 'Additional duration of the last frame', minimum = 00, maximum = 10, step = 1, value = 0)
            with gr.Row():
                debug = gr.Checkbox(label = 'Debug info', value = False)
                run_incomplete = gr.Checkbox(label = 'Run on incomplete', value = True)
                tmp_delete = gr.Checkbox(label = 'Delete intermediate', value = True)
                out_create = gr.Checkbox(label = 'Create animation', value = True)
            with gr.Row():
                tmp_path = gr.Textbox(label = 'Intermediate files path', lines = 1, value = 'intermediate')
                out_path = gr.Textbox(label = 'Output animation path', lines = 1, value = 'animation')

        debug.change(fn=lambda: log.setLevel(logging.DEBUG) if debug.value else log.setLevel(logging.INFO))
        return [is_enabled, codec, interpolation, duration, skip_steps, last_frame_duration, debug, run_incomplete, tmp_delete, out_create, tmp_path, out_path]


    # runs on each step for always-visible scripts
    def process(self, p, is_enabled, codec, interpolation, duration, skip_steps, last_frame_duration, debug, run_incomplete, tmp_delete, out_create, tmp_path, out_path): # pylint: disable=arguments-differ, unused-argument
        if is_enabled:
            # save original callback
            global orig_callback_state # pylint: disable=global-statement
            if orig_callback_state == 'undefined':
                log.debug(f'Steps animation patching sampler callback for: {p.sampler_name}')
                if p.sampler_name in ['DDIM', 'PLMS', 'UniPC']:
                    orig_callback_state = VanillaStableDiffusionSampler.update_step
                else:
                    orig_callback_state = KDiffusionSampler.callback_state

            # set preview mode to full so interim images have full resolution
            if shared.opts.data['show_progress_type'] != 'Full':
                global current_preview_mode # pylint: disable=global-statement
                current_preview_mode = shared.opts.data['show_progress_type']
                log.info(f"Steps animation setting preview type to Full (current {shared.opts.data['show_progress_type']})")
                shared.opts.data['show_progress_type'] = 'Full'

            # define custom callback
            def callback_state(self, d):
                res = orig_callback_state(self, d) # execute original callback
                global current_step # pylint: disable=global-statement
                current_step = shared.state.sampling_step + 1
                if p.sampler_name in ['DDIM', 'PLMS', 'UniPC']:
                    current_step -= 1
                for batch in range(0, p.batch_size):
                    index = p.iteration * p.batch_size + batch
                    fn = f"{p.iteration:02d}{batch:02d}{shared.state.sampling_step:03d}-{str(p.all_seeds[index])}-{safestring(p.all_prompts[index])[:name_length]}"
                    ext = shared.opts.data['samples_format']
                    if (skip_steps == 0) or (current_step > skip_steps):
                        log.debug(f'Steps animation saving interim image: step={current_step} batch={batch} iteration={p.iteration}: {fn}.{ext}')
                        try:
                            # latent = d['denoised'] if 'denoised' in d else d # shared.state.current_latent
                            latent = d if isinstance(d, torch.Tensor) else d['denoised'] # shared.state.current_latent
                            image = sample_to_image(samples = latent, index = batch % p.batch_size)
                            infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments=[], position_in_batch=batch, iteration=p.iteration)
                            infotext = f"{infotext}, intermediate: {current_step:03d}"
                            inpath = os.path.join(p.outpath_samples, tmp_path)
                            save_image(image, inpath, '', extension = ext, short_filename = False, no_prompt = True, forced_filename = fn, info = infotext)
                            temp_files.append(f'{fn}.{ext}')
                        except Exception as e:
                            log.error(f'Steps animation error: save intermediate image: {e}')
                return res

            # set custom callback
            if orig_callback_state != 'undefined':
                if p.sampler_name in ['DDIM', 'PLMS', 'UniPC']:
                    setattr(VanillaStableDiffusionSampler, 'update_step', callback_state)
                else:
                    setattr(KDiffusionSampler, 'callback_state', callback_state)


    # run at the end of sequence for always-visible scripts
    def postprocess(self, p, processed, is_enabled, codec, interpolation, duration, skip_steps, last_frame_duration, debug, run_incomplete, tmp_delete, out_create, tmp_path, out_path):  # pylint: disable=arguments-differ

        def exec_cmd(cmd: string, debug: bool = False):
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True, env=os.environ, check=False)
            l = log.error if result.returncode != 0 else log.debug
            orig_level = log.getEffectiveLevel()
            log.setLevel(logging.DEBUG if debug else logging.INFO)
            l(f'Steps animation: command={cmd} returncode={result.returncode}')
            if len(result.stdout) > 0:
                l(f'Steps animation output: {result.stdout}')
            if len(result.stderr) > 0:
                l(f'Steps animation output: {result.stderr}')
            log.setLevel(orig_level)
            return result.stdout if result.returncode == 0 else result.stderr

        def check_codec(codec: string, debug: bool = False):  # pylint: disable=unused-argument
            ffmpeg = find_ffmpeg_binary()
            cmd = f"{ffmpeg} -hide_banner -encoders"
            stdout = exec_cmd(cmd, debug=False)
            lines = stdout.splitlines()
            lines = [line.strip() for line in lines if line.strip().startswith('V') and '=' not in line]
            codecs = [line.split()[1] for line in lines]
            return codec in codecs

        def unique_filename(basename:str, ext:str) -> str:
            filename = f'{basename}.{ext}'
            if os.path.isfile(filename):
                index = 1
                while True:
                    filename = f'{basename}-{str(index)}.{ext}'
                    if not os.path.isfile(filename):
                        break
                    index += 1
            return filename

        # restore sampler callback
        global orig_callback_state # pylint: disable=global-statement
        if orig_callback_state != 'undefined':
            log.debug(f'Steps animation restoring sampler callback for: {p.sampler_name}')
            if p.sampler_name in ['DDIM', 'PLMS', 'UniPC']:
                VanillaStableDiffusionSampler.update_step = orig_callback_state
            else:
                KDiffusionSampler.callback_state = orig_callback_state
            orig_callback_state = 'undefined'

        # reset preview mode
        if current_preview_mode != 'undefined':
            shared.opts.data['show_progress_type'] = current_preview_mode

        if not is_enabled:
            return

        # callback was never initiated
        global current_step # pylint: disable=global-statement
        if current_step == 0:
            log.error('Steps animation error: steps is zero, likely using unsupported sampler or interrupted')
            return
        # callback happened too early, it happens with large number of steps and some samplers or if interrupted
        if vars(processed)['steps'] < vars(processed)['steps']:
            log.debug(f'Steps animation warning: postprocess early call: current={vars(processed)["steps"]} target={vars(processed)["steps"]}')
            if not run_incomplete:
                return
        # create dictionary with all input and output parameters
        v = vars(processed)
        params = {
            'prompt': safestring(v['prompt']),
            'negative': safestring(v['negative_prompt']),
            'seed': 0, # will be set later
            'sampler': v['sampler_name'],
            'cfgscale': v['cfg_scale'],
            'steps': v['steps'],
            'laststep': current_step,
            'skip': skip_steps,
            'batchsize': v['batch_size'],
            'batchcount': p.n_iter,
            'info': safestring(v['info']),
            'model': v['info'].split('Model:')[1].split()[0] if ('Model:' in v['info']) else 'unknown', # parse string if model info is present
            'embedding': v['info'].split('Used embeddings:')[1].split()[0] if ('Used embeddings:' in v['info']) else 'none',  # parse string if embedding info is present
            'faces': v['face_restoration_model'],
            'timestamp': v['job_timestamp'],
            'inpath': os.path.join(p.outpath_samples, tmp_path),
            'outpath': os.path.join(p.outpath_samples, out_path),
            'codec': 'lib' + codec,
            'duration': duration, #- last_frame_duration,
            'last_frame_duration': last_frame_duration,
            'interpolation': interpolation,
            'loglevel': 'error',
            'cli': cli_template,
            'framerate': max(0, 1.0 * (current_step - skip_steps) / duration), #(duration - last_frame_duration)),
            'videorate': video_rate,
            'author': author,
            'preset': presets[codec],
            'extension': shared.opts.data['samples_format'],
            'short_name': '', # will be set later
            'flags': '-movflags +faststart',
            'ffmpeg': find_ffmpeg_binary(), # detect if ffmpeg executable is present in path
            'ffprobe': shutil.which('ffprobe'), # detect if ffmpeg executable is present in path
        }
        # append conditionals to dictionary
        vfilters = ''
        params['minterpolate'] = '' if (params['interpolation'] == 'none') else f'minterpolate=mi_mode={params["interpolation"]},fifo'
        params['tpad'] = '' if params['last_frame_duration'] == 0 else f'tpad=stop_mode=clone:stop_duration={params["last_frame_duration"]}'
        if params['minterpolate'] != '' or params['tpad'] != '':
            vfilters = '-vf '
        if params['minterpolate'] != '':
            vfilters = vfilters + params['minterpolate']
        if params['tpad'] != '':
            if params['minterpolate'] != '':
                vfilters = vfilters + ',' + params['tpad']
            else:
                vfilters = vfilters + params['tpad']
        params['vfilters'] = vfilters

        if params['codec'] == 'libvpx-vp9':
            suffix = '.webm'
        elif params['codec'] == 'libprores_ks':
            suffix = '.mov'
        else:
            suffix = '.mp4'
        for iteration in range(0, params['batchcount']):
            for batch in range(0, params['batchsize']):
                index = iteration * p.batch_size + batch
                log.debug(f'Steps animation processing batch={batch + 1}/{params["batchsize"]} iteration={iteration + 1}/{params["batchcount"]}')
                params['seed'] = v['all_seeds'][index]
                params['prompt'] = safestring(v['all_prompts'][index])
                params['short_name'] = str(params['seed']) + '-' + params['prompt'][:name_length]
                params['outfile'] = unique_filename(os.path.join(params['outpath'], params['short_name']), suffix)
                params['sequence'] = f'{iteration:02d}{batch:02d}{(skip_steps + 1):03d}'
                params['description'] = '{prompt} | negative {negative} | seed {seed} | sampler {sampler} | cfgscale {cfgscale} | steps {steps} | last {laststep} | model {model} | embedding {embedding} | faces {faces} | timestamp {timestamp} | interpolation {interpolation}'.format(**params)
                current_step = 0 # reset back to zero
                if debug:
                    params['loglevel'] = 'info'
                    log.debug(f'Steps animation params: {json.dumps(params, indent = 2)}')
                if out_create:
                    imgs = [f for f in os.listdir(params['inpath']) if f.startswith(f'{iteration:02d}{batch:02d}') and params['short_name'][:16] in f]
                    if params['framerate'] == 0:
                        log.error('Steps animation error: framerate is zero')
                        return
                    if len(imgs) == 0:
                        log.error('Steps animation no interim images were found')
                        return
                    if not os.path.isdir(params['outpath']):
                        log.info(f'Steps animation create folder: {params["outpath"]}')
                        pathlib.Path(params['outpath']).mkdir(parents=True, exist_ok=True)
                    if not os.path.isdir(params['inpath']) or not os.path.isdir(params['outpath']):
                        log.error(f'Steps animation error: folder not found: {params["inpath"], params["outpath"]}')
                        return

                    if params['ffmpeg'] is None:
                        log.error('Steps animation error: ffmpeg not found:')
                    elif not check_codec(params['codec'], debug):
                        log.error(f"Steps animation error: codec {params['codec']} not supported by ffmpeg")
                    else:
                        log.info(f'Steps animation creating movie sequence: {params["outfile"]} images={len(imgs)}')
                        log.debug(f'Steps animation processing batch: {batch}')
                        log.debug(f'Steps animation using images: {imgs}')
                        cmd = params['cli'].format(**params)
                        # actual ffmpeg call
                        exec_cmd(cmd, debug)

                    if debug:
                        if params['ffprobe'] is None:
                            log.error('Steps animation verify error: ffprobe not found')
                        else:
                            probe = f"ffprobe -hide_banner -print_format json -show_streams \"{params['outfile']}\""
                            exec_cmd(probe, debug)

        if tmp_delete:
            log.debug(f'Steps animation removing {len(temp_files)} files from temp folder: {params["inpath"]}')
            for temp_file in temp_files:
                for fn in pathlib.Path(params['inpath']).glob(os.path.basename(os.path.splitext(temp_file)[0]) + '*'):
                    fn.unlink()

        temp_files.clear()
