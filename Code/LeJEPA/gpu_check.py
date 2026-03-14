import torch
print(f"GPU 가용 여부: {torch.cuda.is_available()}")
print(f"현재 장치 번호: {torch.cuda.current_device()}")
print(f"GPU 이름: {torch.cuda.get_device_name(0)}")