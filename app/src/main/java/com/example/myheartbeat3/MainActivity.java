package com.example.myheartbeat3;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.os.Bundle;
import android.annotation.TargetApi;
import android.content.pm.PackageManager;
import android.os.Build;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.dnn.Net;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

import static android.Manifest.permission.CAMERA;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "opencv";
    private Mat matInput;
    private Mat matResult;
    private String proto;
    private String weights;
    private String haar_model;

    private Net net;

    private CameraBridgeViewBase mOpenCvCameraView;

    public native void main(long matAddrInput, long matAddrResult, long c_time);
    public native void test(String proto, String weights, String haar_model);
    public native void setRPPG(int w, int h);
    public native void setmyisset(double mean, double std);

    public native boolean getlove();
    public native double getstd();
    public native double getmean();
    public native double getchecksig();

    //public native void deep(long matAddrInput, long matAddrResult);

    double mean;
    double std;

    double this_std = -1;
    double this_mean = -1;

    ImageButton heartbutton;
    Button barbutton;

    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }

    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        try {
            // Read data from assets.
            BufferedInputStream inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        Intent intent = getIntent(); /*????????? ??????*/

        mean = intent.getExtras().getDouble("mean");
        std = intent.getExtras().getDouble("std");

        heartbutton = findViewById(R.id.heart_button);
        barbutton = findViewById(R.id.bar_button);

        Log.d("is_get_mean?", "mean:" + mean);
        Log.d("is_get_mean?", "std:" + std);

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(1); // front-camera(1),  back-camera(0)
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        proto = getPath("deploy.prototxt", this);
        weights = getPath("res10_300x300_ssd_iter_140000.caffemodel", this);

        haar_model = getPath("haarcascade_frontalface_alt.xml", this);
        //net = Dnn.readNetFromCaffe(proto, weights);

        test(proto, weights, haar_model);
        setRPPG(width, height);
        setmyisset(mean, std);

        Log.i(TAG, "Network loaded successfully");
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        matInput = inputFrame.rgba();

        if ( matResult == null ) {
            matResult = new Mat(matInput.rows(), matInput.cols(), matInput.type());
        }

        //Core.rotate(matInput, matInput, Core.ROTATE_90_CLOCKWISE); //Core.ROTATE_180); //ROTATE_180 or ROTATE_90_COUNTERCLOCKWISE

//        Mat resize300 = new Mat();
//        Imgproc.resize(frameRGB, resize300, new Size(300, 300));
//
//        Mat blob = Dnn.blobFromImage(resize300, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0));
//        net.setInput(blob);
//        Mat detection = net.forward();
//
//        Log.d(TAG, detection.toString());

        //deep(matInput.getNativeObjAddr(), matResult.getNativeObjAddr());

        long time = System.currentTimeMillis();
        Log.d(TAG, "time java : "+time);

        main(matInput.getNativeObjAddr(), matResult.getNativeObjAddr(), time);
        //Core.rotate(matResult, matResult, Core.ROTATE_90_COUNTERCLOCKWISE);

        this_std = getstd();
        this_mean = getmean();

        runOnUiThread(new Runnable() {
            public void run() {
                barbutton.setText(String.format("?????? ??????:%.1f ?????? ????????????:%.2f\n5?????? ?????? ??????:%.1f 5?????? ?????? ????????????:%.2f t-?????????:%.2f", mean, std, this_mean, this_std, getchecksig()));
            }
        });

        boolean islove = getlove();
        Log.d("love check", "islove : "+islove);

        if(!islove){
            runOnUiThread(new Runnable() {
                public void run() {
                    heartbutton.setImageResource(R.drawable.heart_no_love);
                }
            });
        }
        else{
            runOnUiThread(new Runnable() {
                public void run() {
                    heartbutton.setImageResource(R.drawable.heart_love);
                }
            });
        }

        return matResult;
    }


    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }


    //??????????????? ????????? ?????? ?????????
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;


    protected void onCameraPermissionGranted() {
                List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
                if (cameraViews == null) {
            return;
        }
        for (CameraBridgeViewBase cameraBridgeViewBase: cameraViews) {
            if (cameraBridgeViewBase != null) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            }
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        boolean havePermission = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
                havePermission = false;
            }
        }
        if (havePermission) {
            onCameraPermissionGranted();
        }
    }

    @Override
    @TargetApi(Build.VERSION_CODES.M)
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            onCameraPermissionGranted();
        }else{
            showDialogForPermission("?????? ??????????????? ???????????? ????????????????????????.");
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }


    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {

        AlertDialog.Builder builder = new AlertDialog.Builder( MainActivity.this);
        builder.setTitle("??????");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("???", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("?????????", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }


}