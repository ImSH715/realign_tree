from PIL import Image
import os

# 초대용량 TIF 파일을 열 때 발생하는 보안 에러 방지
Image.MAX_IMAGE_PIXELS = None 

def crop_specific_area_lossless(input_path, output_path, left, upper, right, lower):
    """
    거대한 TIF 이미지에서 원하는 구역만 원본 화질/채널 손실 없이 잘라내어 TIF로 저장합니다.
    """
    print(f"\n[단일 크롭] '{input_path}' 이미지 불러오는 중...")
    try:
        with Image.open(input_path) as img:
            print(f"-> 원본 해상도: {img.size[0]} x {img.size[1]}")
            print(f"-> 원본 색상 모드: {img.mode} (이 모드를 그대로 유지합니다)")
            
            # 지정된 박스 좌표로 자르기
            box = (left, upper, right, lower)
            print(f"-> 지정된 영역 {box} (크기: {right-left}x{lower-upper}) 자르는 중...")
            
            # 원본 화질 그대로 크롭
            cropped_img = img.crop(box)
            
            # 압축 없이(또는 무손실 압축으로) 원본 퀄리티 그대로 TIF 저장
            # save_all=True 속성을 주면 다중 페이지 TIF일 경우에도 대비할 수 있습니다.
            cropped_img.save(output_path, format="TIFF", compression=None)
            print(f"-> [완료] '{output_path}'로 원본 화질 그대로 성공적으로 저장되었습니다!")
            
    except Exception as e:
        print(f"에러 발생: {e}")

def create_training_patches_lossless(input_path, output_dir, patch_size=224):
    """
    거대한 영토 이미지를 동일한 퀄리티의 수많은 .tif 조각들로 분할합니다.
    """
    print(f"\n[그리드 크롭] '{input_path}'를 {patch_size}x{patch_size} 조각으로 분할합니다...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            count = 0
            
            for y in range(0, height - patch_size + 1, patch_size):
                for x in range(0, width - patch_size + 1, patch_size):
                    box = (x, y, x + patch_size, y + patch_size)
                    patch = img.crop(box)
                    
                    # 확장자를 .tif로 지정하여 저장
                    patch_filename = os.path.join(output_dir, f"patch_y{y}_x{x}.tif")
                    patch.save(patch_filename, format="TIFF", compression=None)
                    count += 1
                    
            print(f"-> [완료] 총 {count}개의 무손실 .tif 패치가 '{output_dir}'에 저장되었습니다!")
            
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    # ==========================================
    # 사용자 설정 영역
    # ==========================================
    INPUT_FILE = image_path = r"G:\.shortcut-targets-by-id\1IWblie-cf89tMuc4dQ8umWTHw7XOdl0p\PROYECTO FORGE\01. Ortomosaicos\2023\2023-01\25-PUC-C-DE-CPC-002-12_18032023_001_idw_transparent_mosaic_group1.tif"
    
    OUTPUT_FILE = "cropped_tif/cropped_target_area.tif"
    
    # ------------------------------------------
    # 1. 특정 구역만 잘라내기
    # ------------------------------------------
    crop_specific_area_lossless(
        input_path=INPUT_FILE, 
        output_path=OUTPUT_FILE,
        left=500, 
        upper=2500, 
        right=3500, 
        lower=4500
    )
    
    # ------------------------------------------
    # 2. 전체 영토를 여러 개의 .tif 타일로 쪼개기 (필요시 주석 해제)
    # ------------------------------------------
    # create_training_patches_lossless(
    #     input_path=INPUT_FILE, 
    #     output_dir="dataset_patches_tif", 
    #     patch_size=1024  # 예: 1024x1024 크기로 크게 자르기
    # )