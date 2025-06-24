from ultralytics import YOLO
import torchvision
import torch


def main():
    print(torch.cuda.is_available())  # True면 GPU 사용 가능
    print(torch.cuda.get_device_name(0))  # GPU 이름 확인
    print(torchvision.ops.nms)

    model = YOLO("usingModel/last.pt")
    model.to('cuda')

    results = model.train(
        data="data.yaml",
        epochs=50,
        batch=16,         # 4~8 추천 (너무 크면 CUDA OOM 에러 발생)
        imgsz=640,       # 해상도는 기본 640이 적당
        workers=2,       # GPU가 약한 경우에는 0~2 추천
        device=0
    )

    model.val()
    model.export(format="onnx", dynamic=True)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()