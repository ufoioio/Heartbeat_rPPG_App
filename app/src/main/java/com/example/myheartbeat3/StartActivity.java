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
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.dnn.Net;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

import static android.Manifest.permission.CAMERA;


public class StartActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "opencv";
    private Mat matInput;
    private Mat matResult;
    private String proto;
    private String weights;
    private String haar_model;

    private Net net;

    private CameraBridgeViewBase mOpenCvCameraView;

    double std;
    double mean;
    double checked_std = -1;
    double checked_mean = -1;
    boolean ischecked = false;

    Button startbutton;
    Button reset_button;

    Button tv;


    public native void main(long matAddrInput, long matAddrResult, long c_time);
    public native void test(String proto, String weights, String haar_model);
    public native void setRPPG(int w, int h);

    public native double sendmean();
    public native double sendstd();
    public native double getstd();
    public native double getmean();

    //public native void deep(long matAddrInput, long matAddrResult);

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

        setContentView(R.layout.activity_start);

        startbutton = findViewById(R.id.start_button);
        reset_button = findViewById(R.id.start_reset_button);

        tv = findViewById(R.id.start_tv);

        startbutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                //double my_mean_set = sendmean();
                Log.d("is_get_mean?", "startactivity_mean:" + checked_mean);
                intent.putExtra("mean", checked_mean);

                //double my_std_set = sendstd();
                Log.d("is_get_mean?", "startactivity_std:" + checked_std);
                intent.putExtra("std", checked_std);

                startActivity(intent);
            }
        });

        reset_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ischecked = false;
            }
        });


        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.startactivity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(1); // front-camera(1),  back-camera(0)

        //mOpenCvCameraView.setMaxFrameSize(2000, 800);
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

        std = getstd();
        mean = getmean();
        Log.d("start_std", "std:" + std);

        if(!ischecked){
            if(std < 2 && std > 0){
                ischecked = true;
                checked_std = std;
                checked_mean = mean;
            }
            else{
                runOnUiThread(new Runnable() {
                    public void run() {
                        tv.setBackgroundColor(Color.RED);
                        tv.setText(String.format("아직 기준 맥박이 측정되지 않았습니다.\n(std:%.2f mean:%.1f)", std, mean));
                        startbutton.setVisibility(View.INVISIBLE);
                        reset_button.setVisibility(View.INVISIBLE);
                    }
                });
            }
        }
        else{
            runOnUiThread(new Runnable() {
                public void run() {
                    tv.setBackgroundColor(Color.GREEN);
                    tv.setText(String.format("기준 맥박이 측정되었습니다.\n버튼을 눌러주세요!\n(std:%.2f mean:%.1f)", checked_std, checked_mean));
                    startbutton.setVisibility(View.VISIBLE);
                    reset_button.setVisibility(View.VISIBLE);
                }
            });
        }
        //Core.rotate(matResult, matResult, Core.ROTATE_90_CLOCKWISE);

        return matResult;
    }


    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }


    //여기서부턴 퍼미션 관련 메소드
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
            showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }


    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {

        AlertDialog.Builder builder = new AlertDialog.Builder( StartActivity.this);
        builder.setTitle("알림");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }
}