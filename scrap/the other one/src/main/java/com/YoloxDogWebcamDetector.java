package com;

import ai.onnxruntime.*;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class YoloxDogWebcamDetector {

    static final int INPUT_SIZE = 640;
    static final int DOG_CLASS_ID = 16;
    static final float CONF_THRESHOLD = 0.3f;
    static final float NMS_IOU_THRESHOLD = 0.3f;

    static { OpenCV.loadLocally(); }

    public static void main(String[] args) throws Exception {

        String modelPath = "yolox_l.onnx";

        // ONNX session
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, options);

        // Open webcam
        VideoCapture cap = new VideoCapture(0); // 0 = default camera
        if (!cap.isOpened()) throw new RuntimeException("Cannot open camera");

        Mat frame = new Mat();

        System.out.println("Press ESC to exit...");

        while (true) {
            if (!cap.read(frame)) {
                System.out.println("No frame captured");
                break;
            }

            // Convert OpenCV Mat to BufferedImage
            BufferedImage buffered = matToBufferedImage(frame);

            // Preprocess for YOLOX
            PreprocessResult prep = preprocess(buffered);

            // Create ONNX tensor
            long[] shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(prep.data), shape);

            // Run inference
            String inputName = session.getInputNames().iterator().next();
            OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));
            float[][][] output = (float[][][]) result.get(0).getValue();

            // Decode detections, apply NMS, draw boxes
            decodeAndDrawVideo(output, prep, buffered);

            // Display frame
            Mat outFrame = bufferedImageToMat(buffered);
            HighGui.imshow("Dog Detection", outFrame);
            int key = HighGui.waitKey(1);
            if (key == 27) break; // ESC to exit
        }

        cap.release();
        session.close();
        env.close();
        HighGui.destroyAllWindows();
    }

    // --- Preprocess ---
    static class PreprocessResult {
        float[] data;
        float scale, xOffset, yOffset;
        int origW, origH;
    }

    static PreprocessResult preprocess(BufferedImage img) {
        int origW = img.getWidth(), origH = img.getHeight();
        float scale = Math.min((float)INPUT_SIZE / origW, (float)INPUT_SIZE / origH);
        int newW = Math.round(origW * scale), newH = Math.round(origH * scale);

        BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0,0,newW,newH,null);
        g.dispose();

        BufferedImage padded = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = padded.createGraphics();
        g2.setColor(new Color(114,114,114));
        g2.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);

        float xOffset = (INPUT_SIZE - newW)/2f;
        float yOffset = (INPUT_SIZE - newH)/2f;
        g2.drawImage(resized, Math.round(xOffset), Math.round(yOffset), null);
        g2.dispose();

        float[] data = new float[3*INPUT_SIZE*INPUT_SIZE];
        int rIndex=0, gIndex=INPUT_SIZE*INPUT_SIZE, bIndex=2*INPUT_SIZE*INPUT_SIZE;
        float[] mean={0.485f,0.456f,0.406f}, std={0.229f,0.224f,0.225f};

        for (int y=0;y<INPUT_SIZE;y++) {
            for (int x=0;x<INPUT_SIZE;x++) {
                int rgb=padded.getRGB(x,y);
                float r = ((rgb>>16)&0xFF)/255f;
                float gC = ((rgb>>8)&0xFF)/255f;
                float b = (rgb&0xFF)/255f;

                r = (r-mean[0])/std[0];
                gC = (gC-mean[1])/std[1];
                b = (b-mean[2])/std[2];

                data[rIndex++]=r; data[gIndex++]=gC; data[bIndex++]=b;
            }
        }

        PreprocessResult res = new PreprocessResult();
        res.data = data; res.scale = scale; res.xOffset = xOffset; res.yOffset = yOffset;
        res.origW = origW; res.origH = origH;
        return res;
    }

    // --- Mat <-> BufferedImage conversion ---
    static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage image = new BufferedImage(mat.width(), mat.height(), type);
        mat.get(0,0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
        return image;
    }

    static Mat bufferedImageToMat(BufferedImage img) {
        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(img.getHeight(), img.getWidth(), CvType.CV_8UC3);
        mat.put(0,0,pixels);
        return mat;
    }

    // --- NMS + Detection ---
    static class Detection {
        float x1,y1,x2,y2,conf;
        Detection(float x1,float y1,float x2,float y2,float conf){this.x1=x1;this.y1=y1;this.x2=x2;this.y2=y2;this.conf=conf;}
    }

    static float iou(Detection a, Detection b){
        float xx1=Math.max(a.x1,b.x1),yy1=Math.max(a.y1,b.y1),xx2=Math.min(a.x2,b.x2),yy2=Math.min(a.y2,b.y2);
        float w=Math.max(0,xx2-xx1),h=Math.max(0,yy2-yy1);
        float inter=w*h,areaA=(a.x2-a.x1)*(a.y2-a.y1),areaB=(b.x2-b.x1)*(b.y2-b.y1);
        return inter/(areaA+areaB-inter);
    }

    static List<Detection> nms(List<Detection> boxes,float iouThreshold){
        boxes.sort((a,b)->Float.compare(b.conf,a.conf));
        List<Detection> res=new ArrayList<>();
        while(!boxes.isEmpty()){
            Detection best=boxes.remove(0);
            res.add(best);
            boxes.removeIf(b->iou(best,b)>iouThreshold);
        }
        return res;
    }

    // --- Decode + Draw ---
    static void decodeAndDrawVideo(float[][][] output, PreprocessResult prep, BufferedImage image){
        float[][] predictions=output[0];
        int[] strides={8,16,32};
        int[][] sizes={{80,80},{40,40},{20,20}};
        List<Detection> allDetections=new ArrayList<>();
        int index=0;

        for(int s=0;s<3;s++){
            int stride=strides[s], gw=sizes[s][0], gh=sizes[s][1];
            for(int gy=0;gy<gh;gy++){
                for(int gx=0;gx<gw;gx++){
                    if(index>=predictions.length) break;
                    float[] det=predictions[index++];
                    float tx=det[0],ty=det[1],tw=det[2],th=det[3];
                    float predX=(tx+gx)*stride,predY=(ty+gy)*stride,predW=(float)Math.exp(tw)*stride,predH=(float)Math.exp(th)*stride;
                   // float obj=det[4], dogProb=det[5+DOG_CLASS_ID], conf=obj*dogProb;

                    float obj=det[4], dogProb=det[5+DOG_CLASS_ID], conf=dogProb;
                    if(conf>CONF_THRESHOLD){
                        float x1=(predX-predW/2-prep.xOffset)/prep.scale;
                        float y1=(predY-predH/2-prep.yOffset)/prep.scale;
                        float x2=(predX+predW/2-prep.xOffset)/prep.scale;
                        float y2=(predY+predH/2-prep.yOffset)/prep.scale;
                        allDetections.add(new Detection(x1,y1,x2,y2,conf));
                    }
                }
            }
        }

        List<Detection> finalDetections=nms(allDetections,NMS_IOU_THRESHOLD);
        Graphics2D g=image.createGraphics();
        g.setStroke(new BasicStroke(2));
        g.setColor(Color.RED);
        g.setFont(new Font("Arial",Font.BOLD,16));
        for(Detection det:finalDetections){
            float dx1=det.x1*prep.scale+prep.xOffset, dy1=det.y1*prep.scale+prep.yOffset;
            float dx2=det.x2*prep.scale+prep.xOffset, dy2=det.y2*prep.scale+prep.yOffset;
            g.drawRect(Math.round(dx1),Math.round(dy1),Math.round(dx2-dx1),Math.round(dy2-dy1));
            g.drawString(String.format("Dog: %.2f",det.conf),Math.round(dx1),Math.max(0,Math.round(dy1-5)));
        }
        g.dispose();
    }
}
