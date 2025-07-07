package com.example.damo_yolo_test;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.BufferedInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import ai.onnxruntime.OrtException;

public class MainActivity extends AppCompatActivity {
    DAMOYOLO damoyolo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        findViewById(R.id.button_first).setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, 1);
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            ImageView imgView = findViewById(R.id.imageView);
//            TextView result = binding.result;
            Bitmap bitmap;

            try {
                BufferedInputStream inputStream = new BufferedInputStream(
                        this.getContentResolver().openInputStream(data.getData())
                );
                bitmap = BitmapFactory.decodeStream(inputStream);
            } catch (FileNotFoundException e) {
                Log.e("onActivityResult", "error", e);

                AlertDialog.Builder builder = new AlertDialog.Builder(this);
                builder.setTitle("Error");
                builder.setMessage("File not found");
                builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                    }
                });
                AlertDialog dialog = builder.create();
                dialog.show();

                return;
            }

            if (!BitmapValidator.isValidBitmap(bitmap)) {
                Log.e("onActivityResult", "error", new Error("Invalid image file"));

                AlertDialog.Builder builder = new AlertDialog.Builder(this);
                builder.setTitle("Error");
                builder.setMessage("Invalid image file");
                builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                    }
                });
                AlertDialog dialog = builder.create();
                dialog.show();

                return;
            }

            imgView.setImageBitmap(bitmap);

            new Thread(() -> {
                execute(bitmap);
            }).start();
        }
    }

    private void execute(Bitmap origImage) {
        List<String> labels = Arrays.asList("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush");

        if (damoyolo == null) {
            byte[] modelBytes;
            try {
                InputStream in = this.getAssets().open("damoyolo_tinynasL45_L_519.onnx");
                modelBytes = new byte[in.available()];
                in.read(modelBytes);
                in.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            try {
                damoyolo = new DAMOYOLO(modelBytes);
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }

        try {
            List<DAMOYOLO.Detection> detectionList = damoyolo.run(origImage, 0.4f, 0.45f);
            Bitmap drawed = DAMOYOLO.drawDetections(origImage, detectionList, labels);

            new Handler(Looper.getMainLooper()).post(() -> {
                ImageView imageView = findViewById(R.id.imageView);
                imageView.setImageBitmap(drawed);
            });
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }
}