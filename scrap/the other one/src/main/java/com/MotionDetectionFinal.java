package com;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;

public class MotionDetectionFinal {

    public static void main(String[] args) throws Exception {

        String videoPath = "dogi2.mp4";

        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoPath);
        grabber.start();

        // Get FPS for correct playback speed
        double fps = grabber.getFrameRate();
        if (fps <= 0 || fps > 240) fps = 30; // fallback
        long delay = (long)(1000 / fps);

        CanvasFrame originalWindow = new CanvasFrame("Original");
        CanvasFrame motionWindow = new CanvasFrame("Motion Mask");

        originalWindow.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        motionWindow.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);

        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        Mat prevGray = null;

        while (true) {
            Frame frame = grabber.grab();
            if (frame == null) break;

            Mat mat = converter.convert(frame);
            if (mat == null) continue;

            // Resize to stabilize dimensions
            Mat resized = new Mat();
            resize(mat, resized, new Size(640, 480));

            // Convert to grayscale
            Mat gray = new Mat();
            cvtColor(resized, gray, COLOR_BGR2GRAY);

            // Blur to reduce noise
            GaussianBlur(gray, gray, new Size(5, 5), 0);

            if (prevGray != null) {
                // Compute absolute difference
                Mat diff = new Mat();
                absdiff(gray, prevGray, diff);

                // Threshold to get black/white motion mask
                Mat thresh = new Mat();
                threshold(diff, thresh, 25, 255, THRESH_BINARY);

                // Show motion mask
                motionWindow.showImage(converter.convert(thresh));
            }

            // Show original frame
            originalWindow.showImage(converter.convert(resized));

            // Save current frame for next iteration
            prevGray = gray.clone();

            // Keep playback at correct speed
            Thread.sleep(delay);
        }

        grabber.stop();
        originalWindow.dispose();
        motionWindow.dispose();
    }
}
