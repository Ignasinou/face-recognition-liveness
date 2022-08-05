import os
from pathlib import Path
import cv2
import tqdm
from modules import FaceDetection, IdentityVerification, LivenessDetection
import argparse
import subprocess
from centerface import CenterFace
import onnxruntime as ort
from modules import get_faces_from_centerface

sess_options = ort.SessionOptions()
centerface = CenterFace(ort, sess_options, landmarks=True)


root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"

resNet_checkpoint_path = data_folder / "InceptionResnetV1_vggface2.onnx"
facebank_path = data_folder / "hm.csv"

deepPix_checkpoint_path = data_folder / "OULU_Protocol_2_model_0_0.onnx"

parser = argparse.ArgumentParser(description="BigRoom ASD inference")
parser.add_argument('--videoFile')
parser.add_argument('--liveness_th', default=0.03, type=float)
# parser.add_argument('--det_confidence', default=0.2, type=float)
parser.add_argument('--centerface_threshold', type=float, default=0.7, help="centerface th")
parser.add_argument('--det_model', default=1, choices=[0, 1], type=int, help="0 is 2 meters from camera 1 is anywhere")
args = parser.parse_args()

liveness_th = args.liveness_th
min_detection_confidence = args.det_confidence
model_selection = args.det_model

file_extension = "." + args.videoFile.split(".")[-1]
output_video_filename = args.videoFile.replace(file_extension, '_output.mp4')
output_audio_filename = args.videoFile.replace(file_extension, '.wav')

subprocess.call(
    'ffmpeg -i %s -pix_fmt yuv420p -c:v h264_nvenc -cq:a 0 -ac 1 -vn -threads 10 -ar 16000 %s -y' % (
        args.videoFile,
        output_audio_filename),
    shell=True)

#faceDetector = FaceDetection(min_detection_confidence, model_selection)
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(), facebank_path=facebank_path.as_posix())
livenessDetector = LivenessDetection(
    checkpoint_path=deepPix_checkpoint_path.as_posix())

video = cv2.VideoCapture(args.videoFile)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
outVideo = cv2.VideoWriter(
    output_video_filename
    , cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

while True:
    for frame_num in tqdm.tqdm(range(int(num_frames))):
        ret, frame = video.read()

        if not ret:
            break

        #faces, boxes = faceDetector(frame)
        faces, boxes = get_faces_from_centerface(frame,
                                                 centerface_threshold=args.centerface_threshold,
                                                 centerface=centerface)
        if not len(faces):
            outVideo.write(frame.copy())
            continue

        for face_arr, box in zip(faces, boxes):

            min_sim_score, mean_sim_score = identityChecker(face_arr)
            liveness_score = livenessDetector(face_arr)
            if liveness_score >= liveness_th:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            liveness_score = float(f"{liveness_score:0.5f}")
            frame = cv2.rectangle(frame, box[0], box[1], color, 10)
            cv2.putText(frame, f"{liveness_score}", (box[0][0], box[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
        outVideo.write(frame.copy())
    break

outVideo.release()
cv2.destroyAllWindows()
video.release()

subprocess.call(
    'ffmpeg -i %s -i %s -shortest  %s -y' % (output_video_filename,
                                             output_audio_filename,
                                             output_video_filename.replace(file_extension, "_AV.mp4")),
    shell=True)
