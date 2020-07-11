package com.example.testnetdemo;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Module module = null;

        try {
            module = Module.load(assetFilePath(this, "trace_model.pt"));
        } catch (IOException e) {
            Log.e("TestNetDemo", "Error reading assets", e);
        }

        float[] inputTensorArr = new float[112 * 112 * 3];
        Arrays.fill(inputTensorArr, 0.2f);

        Tensor inputTensor = Tensor.fromBlob(inputTensorArr, new long[]{1, 3, 112, 112});


        for(int i = 0; i < 20; i++)
        {
            Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

            float[] outputTensorArr = outputTensor.getDataAsFloatArray();

            String msg = "";
            for(int j = 0; j < 10; j++)
            {
                msg += outputTensorArr[j] + " ";
            }
            Log.d("TestNetDemo", msg);
        }
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
