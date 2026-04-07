package com;

import ai.onnxruntime.*;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Collections;

public class YoloxDogDetector {

    static final int INPUT_SIZE = 640;
    static final int DOG_CLASS_ID = 16; // COCO dog class
    static final float CONF_THRESHOLD = 0.3f; // confidence threshold
    static final float NMS_IOU_THRESHOLD = 0.3f;

    static BufferedImage letterboxedImage;
    static float xOffset, yOffset, scale;
    static int origW, origH;

    public static void main(String[] args) throws Exception {

        String modelPath = "yolox_l.onnx";
        String imagePath = "dog2.jpg";

        float[] inputData = preprocess(imagePath);

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, options);

        long[] shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);
        String inputName = session.getInputNames().iterator().next();

        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));
        float[][][] output = (float[][][]) result.get(0).getValue();

        // Decode, apply NMS, draw boxes
        decodeAndDraw(output, letterboxedImage);

        // Save final image
        ImageIO.write(letterboxedImage, "png", new File("dog_detected.png"));
        System.out.println("Detection result saved as dog_detected.png");

        session.close();
        env.close();
    }

    static float[] preprocess(String imagePath) throws Exception {
        BufferedImage img = ImageIO.read(new File(imagePath));
        if (img == null) throw new RuntimeException("Cannot read image: " + imagePath);

        origW = img.getWidth();
        origH = img.getHeight();

        // Resize with aspect ratio
        scale = Math.min((float) INPUT_SIZE / origW, (float) INPUT_SIZE / origH);
        int newW = Math.round(origW * scale);
        int newH = Math.round(origH * scale);

        BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0, 0, newW, newH, null);
        g.dispose();

        // Letterbox
        letterboxedImage = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = letterboxedImage.createGraphics();
        g2.setColor(new Color(114, 114, 114));
        g2.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

        xOffset = (INPUT_SIZE - newW) / 2f;
        yOffset = (INPUT_SIZE - newH) / 2f;

        g2.drawImage(resized, Math.round(xOffset), Math.round(yOffset), null);
        g2.dispose();

        // Save letterbox image for debugging
        ImageIO.write(letterboxedImage, "png", new File("debug_letterbox.png"));
        System.out.println("Letterboxed image saved as debug_letterbox.png");

        // Convert to CHW float array
        float[] data = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int rIndex = 0, gIndex = INPUT_SIZE * INPUT_SIZE, bIndex = 2 * INPUT_SIZE * INPUT_SIZE;

        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std  = {0.229f, 0.224f, 0.225f};

        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                int rgb = letterboxedImage.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255f;
                float gC = ((rgb >> 8) & 0xFF) / 255f;
                float b = (rgb & 0xFF) / 255f;

                r = (r - mean[0]) / std[0];
                gC = (gC - mean[1]) / std[1];
                b = (b - mean[2]) / std[2];

                data[rIndex++] = r;
                data[gIndex++] = gC;
                data[bIndex++] = b;
            }
        }
        return data;
    }

    // Detection class
    static class Detection {
        float x1, y1, x2, y2, confidence;
        Detection(float x1, float y1, float x2, float y2, float confidence) {
            this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2; this.confidence = confidence;
        }
    }

    static float iou(Detection a, Detection b) {
        float xx1 = Math.max(a.x1, b.x1);
        float yy1 = Math.max(a.y1, b.y1);
        float xx2 = Math.min(a.x2, b.x2);
        float yy2 = Math.min(a.y2, b.y2);

        float w = Math.max(0, xx2 - xx1);
        float h = Math.max(0, yy2 - yy1);
        float inter = w * h;
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (areaA + areaB - inter);
    }

    static List<Detection> nms(List<Detection> boxes, float iouThreshold) {
        boxes.sort(Comparator.comparingDouble(d -> -d.confidence));
        List<Detection> result = new ArrayList<>();
        while (!boxes.isEmpty()) {
            Detection best = boxes.remove(0);
            result.add(best);
            boxes.removeIf(b -> iou(best, b) > iouThreshold);
        }
        return result;
    }

    static void decodeAndDraw(float[][][] output, BufferedImage image) {
        float[][] predictions = output[0];
        int[] strides = {8,16,32};
        int[][] sizes = {{80,80},{40,40},{20,20}};

        List<Detection> allDetections = new ArrayList<>();
        int index = 0;

        for (int s = 0; s < 3; s++) {
            int stride = strides[s];
            int gw = sizes[s][0];
            int gh = sizes[s][1];

            for (int gy = 0; gy < gh; gy++) {
                for (int gx = 0; gx < gw; gx++) {
                    if (index >= predictions.length) break;
                    float[] det = predictions[index++];

                    float tx = det[0], ty = det[1], tw = det[2], th = det[3];
                    float predX = (tx + gx) * stride;
                    float predY = (ty + gy) * stride;
                    float predW = (float)Math.exp(tw) * stride;
                    float predH = (float)Math.exp(th) * stride;

                    float objectness = det[4];
                    float dogProb = det[5 + DOG_CLASS_ID];
                    float confidence = dogProb;

                    if (confidence > CONF_THRESHOLD) {
                        // Convert to original image coordinates
                        float x1 = (predX - predW/2 - xOffset) / scale;
                        float y1 = (predY - predH/2 - yOffset) / scale;
                        float x2 = (predX + predW/2 - xOffset) / scale;
                        float y2 = (predY + predH/2 - yOffset) / scale;
                        allDetections.add(new Detection(x1,y1,x2,y2,confidence));
                    }
                }
            }
        }

        // Apply NMS in original image coordinates
        List<Detection> finalDetections = nms(allDetections, NMS_IOU_THRESHOLD);

        // Draw boxes back on letterboxed image
        Graphics2D g = image.createGraphics();
        g.setStroke(new BasicStroke(2));
        g.setColor(Color.RED);
        g.setFont(new Font("Arial", Font.BOLD, 16));

        for (Detection det : finalDetections) {
            // Convert back to letterbox coordinates for drawing
            float dx1 = det.x1 * scale + xOffset;
            float dy1 = det.y1 * scale + yOffset;
            float dx2 = det.x2 * scale + xOffset;
            float dy2 = det.y2 * scale + yOffset;

            g.drawRect(Math.round(dx1), Math.round(dy1), Math.round(dx2-dx1), Math.round(dy2-dy1));
            String label = String.format("Dog: %.2f", det.confidence);
            g.drawString(label, Math.round(dx1), Math.max(0, Math.round(dy1-5)));
            System.out.println("DOG DETECTED: Conf=" + det.confidence +
                " Box=" + det.x1 + "," + det.y1 + " -> " + det.x2 + "," + det.y2);
        }
        g.dispose();
    }
}
