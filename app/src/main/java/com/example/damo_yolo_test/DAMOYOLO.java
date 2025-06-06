package com.example.damo_yolo_test;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class DAMOYOLO {
    private OrtEnvironment env;
    private OrtSession session;

    public DAMOYOLO(byte[] modelBytes) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        session = env.createSession(modelBytes, opts);
    }
    public static class Detection {
        public float x1, y1, x2, y2, score;
        public int classId;

        public Detection(float x1, float y1, float x2, float y2, float score, int classId) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            this.classId = classId;
        }
    }

    public List<Detection> run(Bitmap bitmap, float scoreTh, float nmsTh) throws OrtException {
        float[][][][] inputData = preprocess(bitmap);

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("images", inputTensor);
        OrtSession.Result result = session.run(inputs);

        float[][][] scores = (float[][][]) result.get(0).getValue();
        float[][][] bboxes = (float[][][]) result.get(1).getValue();

        return postprocess(scores[0], bboxes[0], bitmap.getWidth(), bitmap.getHeight(), scoreTh, nmsTh);
    }

    private float[][][][] preprocess(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true);

        // Onnxでは入力する画像をNCHW形式にする必要がある
        // N: バッチサイズ (画像の数) 1
        // C: チャンネル数 (色の種類) 3
        // H: Height (高さ) 640
        // W: Width (幅) 640
        // [1 3 640 640]
        float[][][][] inputData = new float[1][3][640][640];
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                int pixel = resized.getPixel(x, y);
                float r = Color.red(pixel);
                float g = Color.green(pixel);
                float b = Color.blue(pixel);

                // モデルによってはRGB値を0〜255ではなく0.0〜1.0、-1.0〜1.0に変換する必要があるが、DAMO-YOLOの場合は必要ない？
                // 両者共に試してみたが、元の0〜255の値でしか動作しなかった
                inputData[0][0][y][x] = r;
                inputData[0][1][y][x] = g;
                inputData[0][2][y][x] = b;
            }
        }

        return inputData;
    }

    private List<Detection> postprocess(float[][] scores, float[][] bboxes, int imgW, int imgH, float scoreTh, float nmsTh) {
        float scaleX = (float) imgW / 640;
        float scaleY = (float) imgH / 640;

        List<Detection> detections = new ArrayList<>();
        int numClasses = scores[0].length; // 分類の数を取得
        for (int i = 0; i < bboxes.length; i++) {
            int bestClass = -1;
            float bestScore = 0;
            for (int j = 0; j < numClasses; j++) {
                if (scores[i][j] > bestScore) {
                    bestScore = scores[i][j];
                    bestClass = j;
                }
            }
            if (bestScore > scoreTh) {
                float[] box = bboxes[i];
                float x1 = box[0] * scaleX;
                float y1 = box[1] * scaleY;
                float x2 = box[2] * scaleX;
                float y2 = box[3] * scaleY;
                detections.add(new Detection(x1, y1, x2, y2, bestScore, bestClass));
            }
        }
        return nms(detections, nmsTh);
    }

    private List<Detection> nms(List<Detection> detections, float nmsTh) {
        // NMSを実装
        // TODO: クラスごとでの算出
        // 最も信頼度の高いボックスの抽出するために、スコアで降順に並び替える
        Collections.sort(detections, (a, b) -> Float.compare(b.score, a.score));
        List<Detection> keep = new ArrayList<>();
        for (Detection detection : detections) {
            boolean keepDetection = true;
            for (Detection k : keep) {
                if (iou(detection, k) > nmsTh) {
                    keepDetection = false;
                    break;
                }
            }
            if (keepDetection) {
                keep.add(detection);
            }
        }
        return keep;
    }

    private float iou(Detection a, Detection b) {
        // IoUを実装
        // 2つの領域がどれくらい重なっているかをあらわす指標
        // YOLO系では仕様上重複ありきのため、IoUで重なり具合を算出する
        // https://qiita.com/CM_Koga/items/82d446658957d51836cf
        // https://qiita.com/k-akiyama/items/89714d276871ea339aa9
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

    public static Bitmap drawDetections(Bitmap bitmap, List<Detection> detections/*, List<String> labels*/) {
        Bitmap mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        Canvas canvas = new Canvas(mutable);

        Paint paint = new Paint();
        paint.setStrokeWidth(3);
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.RED);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(30);

        for (Detection detection : detections) {
            canvas.drawRect(new RectF(detection.x1, detection.y1, detection.x2, detection.y2), paint);

            String label = String.valueOf(detection.classId) + String.format(": %.2f", detection.score);
            canvas.drawText(label, detection.x1, detection.y1 - 10, textPaint);
        }

        return mutable;
    }
}
