########################################################
# 유틸
from PIL import ExifTags
from PIL import Image
from PIL import ImageOps

def open_with_exif(filepath):
    image = Image.open(filepath)
    return ImageOps.exif_transpose(image)

#########################################################
# 처리할 이미지 리스트 추리기

from glob import glob

from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

def file_list(base_dir, exts=['jpg', 'png', 'tiff', 'jpeg']):
    
    if base_dir[:2] == './':
        base_dir = base_dir[2:]
    
    # 이미지 리스트를 준비한다.
    # glob은 히든폴더/히든파일은 제외하고 리스팅 한다 
    files = glob(f'{base_dir}/**', recursive=True)
    files = sorted(files)
    
    # 확장자 리스트로 필터링한다.
    EXTS = set([e.upper() for e in exts])
    files = [e for e in files if Path(e).suffix[1:].upper() in EXTS]
    
    # pillow 로 읽어올 수 있는 파일들만 필더링 한다
    # gif 도 버린다
    def readable(e):
        Image.open(e)
        if Path(e).suffix[1:].lower() in ['gif']:
            return False
        try:
            Image.open(e)
            return True
        except:
            return False
        
    imgs = [e for e in tqdm(files) if readable(e)]
    return imgs


#########################################################
# face detect

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import numpy as np
import pandas as pd

# FAN 모듈을 준비한다
import face_alignment
g_fan = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._3D,  # align을 위해 3D를  사용한다.
    face_detector='sfd',    # detector는 SFD를 사용할 것이다.
    face_detector_kwargs=dict(
        path_to_detector=None, # None: FAN 저장소에서 다운로드한다. 
        filter_threshold=0.98  # 명확한 얼굴만 탐지하기 위해 threhold값을 높게 설정한다.
    ))     
    
# 배치 처리를 위한 DataSet을 준비한다
# 고정된 rect_size 의 크기로 ndarray 이미지를 리턴한다
# 정사각형으로 만들기 위해 필요한 경우 오른쪽과 아랫쪽에 패딩을 한다.
# rect_size 이미지에서 원본 사이즈를 얻기 위한 비율도 리턴한다.
class RectDS(Dataset):
    def __init__(self, imgs, img_base_dir =None, rect_size = 512):
        self.imgs = imgs
        self.img_base_dir = img_base_dir
        self.rect_size = rect_size
        
    def __len__(self):
        return len(self.imgs)
    
    def to_rect(im, to_size):
        w, h = im.size
        size = max(w, h)
        r = to_size/size
        w, h = int(w*r), int(h*r)
        im =im.resize((w, h))
        im = np.pad(np.array(im), ((0, to_size-h), (0, to_size-w), (0, 0)), constant_values=128)
        im = im.transpose(2, 0, 1).astype(np.float32)
        return 1/r, im
        
        
    def __getitem__(self, idx):
        path = self.imgs[idx]
        if self.img_base_dir:
            path = Path(self.img_base_dir) / path
        im = open_with_exif(path).convert('RGB')
        r, im = RectDS.to_rect(im, self.rect_size) 
        return str(path), r, np.array(im)
    
def get_landmark(files, batch_size=16, num_workers=0):# landmark 정보를 얻어온다.
    ds = RectDS(files)
    total_paths = []
    total_ratios = []
    total_marks = []
    total_scores = []
    total_boxes = []
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    for i, (paths, ratios, ims) in enumerate(tqdm(dl)):
        marks, scores, boxes = g_fan.get_landmarks_from_batch(ims, return_bboxes=True, return_landmark_score=True)
        total_paths.extend(paths)
        total_ratios.extend(ratios)
        total_marks.extend(marks)
        total_scores.extend(scores)
        total_boxes.extend(boxes)   
    return pd.DataFrame({ 
        'path' : total_paths,
        'ratio' : total_ratios,
        'mark': total_marks,
        'score' : total_scores,
        'box' : total_boxes 
    })


#########################################################
# face crop
from pathlib import Path
import shutil
import pandas as pd
from tqdm.auto import tqdm

def crop_face(row, scale = 0.9, randomness=False):
    im = open_with_exif(row['path']).convert('RGB')
    #display(im.size)
    #display(row)
    idx = np.stack(row['score']).mean(axis=-1).argmax()
    box = np.array(row['box'][idx])*row['ratio'].item()
    box = box.round().astype(np.int32)
    #display('box', box[0])
    (x1, y1, x2, y2) = box[:4]
    
    im = np.array(im)
    c_x, c_y = (x1 + x2)/2, (y1 + y2)/2
    r = scale 
    if randomness:
        r = r-0.05 + 0.1*random.random() 
    #print('r:', r)
    #sz = int(min(r*(x2-x1+1), r*(y2-y1+1), im.shape[0]/2, im.shape[1]/2))
    sz = int(min(max(r*(x2-x1+1), r*(y2-y1+1)), im.shape[0]/2, im.shape[1]/2))
    #sz_t = r*((x2-x1 +1) + (y2-y1+1))/2
    #sz = int(min(sz_t, im.shape[0]/2, im.shape[1]/2))
    x1, y1 = int(max(c_x - sz, 0)), int(max(c_y - sz , 0))
    x2, y2 = x1 + 2*sz, y1 + 2*sz
    im = np.array(im)[y1:y2+1, x1:x2+1, :]
    h, w = im.shape[:2]
    sz = min(h, w)
    im = im[(h-sz)//2:(h-sz)//2 + sz, (w-sz)//2:(w-sz)//2 + sz, :]
    im = Image.fromarray(im)
    #im.thumbnail((512, 512))
    #print(im.size)
    #display(im)
    #src
    return im



def save_croped_face(out_dir, df_face_info, face_resize, scale):
    shutil.rmtree(out_dir, ignore_errors=True)

    errors = []
    oks = []
    
    for idx, row in tqdm(df_face_info.iterrows(), total=len(df_face_info)):
        im = crop_face(row, scale)
        try:
            im = crop_face(row, scale)
        except:
            print('error:', row['path'])
            errors.append(row)
            continue
        dst = Path(out_dir) / '/'.join(row['path'].split('/')[1:])
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
        dst = dst.with_suffix('.jpg')
        im = im.resize((face_resize, face_resize))
        im.save(dst, quality=100, subsampling=0)  
        oks.append(dst)
    return oks
    
    
    
    
    
    