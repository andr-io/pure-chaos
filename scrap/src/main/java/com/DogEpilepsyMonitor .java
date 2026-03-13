package com;

import ai.onnxruntime.*;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Collections;

public class DogEpilepsyMonitor {

    static { OpenCV.loadLocally(); }

    static final int INPUT_SIZE = 320;
    static final float CONF_THRESHOLD = 0.3f;
    static final float KP_THRESHOLD = 0.4f;

    public static void main(String[] args) throws Exception {

        String modelPath = "640_to_320_100.onnx";
        String videoPath = "doggoW.mp4";

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions());

        VideoCapture cap = new VideoCapture(videoPath);
        Mat frame = new Mat();

        while (cap.read(frame)) {

            int originalWidth = frame.cols();
            int originalHeight = frame.rows();

            Mat resized = new Mat();
            Imgproc.resize(frame, resized, new Size(INPUT_SIZE, INPUT_SIZE));

            float[] input = matToCHWFloat(resized);

            OnnxTensor tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(input),
                new long[]{1,3,INPUT_SIZE,INPUT_SIZE}
            );

            OrtSession.Result result = session.run(
                Collections.singletonMap(
                    session.getInputNames().iterator().next(),
                    tensor
                )
            );

            float[][][] output = (float[][][]) result.get(0).getValue();

            drawDetections(frame, output[0], originalWidth, originalHeight);

            HighGui.imshow("Dog Pose", frame);

            if (HighGui.waitKey(1) == 27)
                break;
        }

        cap.release();
        HighGui.destroyAllWindows();
    }

    static float[] matToCHWFloat(Mat img) {

        int channels = 3;
        int width = img.cols();
        int height = img.rows();

        float[] data = new float[channels * width * height];
        int idx = 0;

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {

                    double[] pixel = img.get(y,x);
                    data[idx++] = (float)(pixel[c] / 255.0);
                }
            }
        }

        return data;
    }

    static void drawDetections(Mat frame, float[][] detections,
                               int originalWidth, int originalHeight) {

        float scaleX = (float) originalWidth / INPUT_SIZE;
        float scaleY = (float) originalHeight / INPUT_SIZE;

        for (float[] det : detections) {

            float conf = det[4];
            if (conf < CONF_THRESHOLD) continue;

            int keypointStart = 5;

            for (int k = keypointStart; k + 2 < det.length; k += 3) {

                float x = det[k + 1];
                float y = det[k + 2];
                float kpConf = det[k];

                System.out.println(x +  " " + y + " " + kpConf + "\n");

                if (kpConf < KP_THRESHOLD) continue;

                int px = (int)(x * scaleX);
                int py = (int)(y * scaleY);

                Imgproc.circle(frame,
                    new Point(px, py),
                    4,
                    new Scalar(0,255,0),
                    -1);
            }
        }
    }
}