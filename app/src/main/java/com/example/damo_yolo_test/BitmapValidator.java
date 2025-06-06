package com.example.damo_yolo_test;

import android.graphics.Bitmap;

public class BitmapValidator {
    public static boolean isValidBitmap(Bitmap bitmap) {
        if (bitmap == null) return false;
        if (bitmap.getWidth() <= 0 || bitmap.getHeight() <= 0) return false;
        if (bitmap.isRecycled()) return false;
        if (bitmap.getConfig() == null) return false;

        try {
            bitmap.getPixel(0, 0);
        } catch (IllegalArgumentException e) {
            return false;
        }

        return true;
    }
}
