"""Convert uploaded images/video into a Mip-NeRF-360-style COLMAP dataset."""
import os, re, shutil, subprocess
from pathlib import Path
from PIL import Image

def executable(root, name):
    candidates = ([root/'colmap/build/src/colmap/exe/colmap', root/'colmap/build/colmap']
                  if name == 'colmap' else [])
    for path in candidates:
        if path.is_file() and os.access(str(path), os.X_OK): return str(path)
    found = shutil.which(name)
    if not found: raise RuntimeError('{} executable not found'.format(name))
    return found

def run(command, log, cwd=None, progress=None, progress_range=None):
    log('$ '+' '.join(map(str, command)))
    process = subprocess.Popen(list(map(str, command)), cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    last_fraction = -1.0
    for line in process.stdout:
        line = line.rstrip()
        log(line)
        if progress and progress_range:
            match = re.search(r'\[\s*(\d+)\s*/\s*(\d+)\s*\]', line)
            if match and int(match.group(2)):
                fraction = min(1.0, int(match.group(1)) / int(match.group(2)))
                if fraction > last_fraction:
                    start, end, phase = progress_range
                    progress(start + (end - start) * fraction, phase, line[-120:])
                    last_fraction = fraction
    if process.wait(): raise RuntimeError('Command failed: {}'.format(command[0]))

def registered_image_count(colmap, model):
    result = subprocess.run([colmap, 'model_analyzer', '--path', str(model)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    match = re.search(r'Registered images:\s*(\d+)', result.stdout)
    return int(match.group(1)) if match else 0

def build_dataset(root, source, destination, source_type, frame_count, scales, log,
                  progress=lambda percent, phase, detail='': None):
    root, source, destination = map(Path, (root, source, destination))
    destination.mkdir(parents=True, exist_ok=False)
    input_dir = destination/'input'; input_dir.mkdir()
    progress(1, 'Preparing input', 'Creating dataset workspace')
    if source_type == 'video':
        ffmpeg = executable(root, 'ffmpeg'); ffprobe = executable(root, 'ffprobe')
        duration = float(subprocess.check_output([ffprobe,'-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(source)]))
        fps = max(0.001, int(frame_count)/duration)
        progress(3, 'Extracting video frames', 'Target: {} frames'.format(frame_count))
        run([ffmpeg,'-i',source,'-vf','fps={}'.format(fps),'-frames:v',str(frame_count),input_dir/'frame_%06d.png'], log)
    else:
        extensions={'.png','.jpg','.jpeg','.webp','.tif','.tiff'}
        items = [item for item in source.rglob('*')
                 if item.is_file() and item.suffix.lower() in extensions]
        for index, item in enumerate(items, 1):
            shutil.copy2(str(item), str(input_dir/item.name))
            progress(2 + 6 * index / max(1, len(items)), 'Preparing images',
                     '{} / {} images'.format(index, len(items)))
    colmap=executable(root,'colmap'); distorted=destination/'distorted'; (distorted/'sparse').mkdir(parents=True)
    db=distorted/'database.db'
    progress(8, 'Feature extraction', 'COLMAP is detecting local image features')
    run([colmap,'feature_extractor','--database_path',db,'--image_path',input_dir,'--ImageReader.single_camera','1','--ImageReader.camera_model','OPENCV','--FeatureExtraction.use_gpu','0'],log,
        progress=progress, progress_range=(8, 34, 'Feature extraction'))
    progress(34, 'Feature matching', 'COLMAP is matching image pairs')
    run([colmap,'exhaustive_matcher','--database_path',db,'--FeatureMatching.use_gpu','0'],log,
        progress=progress, progress_range=(34, 60, 'Feature matching'))
    progress(60, 'Sparse reconstruction', 'COLMAP mapper is registering cameras')
    run([colmap,'mapper','--database_path',db,'--image_path',input_dir,'--output_path',distorted/'sparse','--Mapper.ba_global_function_tolerance=0.000001'],log)
    models = [path for path in (distorted/'sparse').iterdir() if path.is_dir()]
    if not models: raise RuntimeError('COLMAP mapper produced no reconstruction')
    ranked = sorted(((registered_image_count(colmap, model), model) for model in models), reverse=True)
    best_count, best_model = ranked[0]
    log('Selecting largest reconstruction: model {} with {} / {} registered images'.format(
        best_model.name, best_count, len(list(input_dir.iterdir()))))
    if best_count < 3: raise RuntimeError('Largest COLMAP reconstruction has fewer than 3 images')
    progress(78, 'Undistorting images', '{} registered cameras'.format(best_count))
    run([colmap,'image_undistorter','--image_path',input_dir,'--input_path',best_model,'--output_path',destination,'--output_type','COLMAP'],log,
        progress=progress, progress_range=(78, 90, 'Undistorting images'))
    sparse=destination/'sparse'; model=sparse/'0'; model.mkdir(exist_ok=True)
    for item in list(sparse.iterdir()):
        if item.name!='0': shutil.move(str(item),str(model/item.name))
    factors = sorted(set(int(x) for x in scales if int(x)>1))
    resize_total = len(factors) * len(list((destination/'images').iterdir()))
    resized = 0
    for factor in factors:
        out=destination/'images_{}'.format(factor); out.mkdir()
        for item in (destination/'images').iterdir():
            with Image.open(item) as image:
                image.resize((max(1,image.width//factor),max(1,image.height//factor)),Image.Resampling.LANCZOS).save(out/item.name)
            resized += 1
            progress(90 + 9 * resized / max(1, resize_total), 'Generating resolutions',
                     '{} / {} resized images'.format(resized, resize_total))
    progress(100, 'Dataset ready', str(destination))
    log('Dataset ready: {}'.format(destination))
