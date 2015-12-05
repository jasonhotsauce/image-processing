package sample;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class Controller {
    // the FXML button
    @FXML
    private Button button;
    // the FXML grayscale checkbox
    @FXML
    private CheckBox grayscale;
    // the FXML logo checkbox
    @FXML
    private CheckBox logoCheckBox;
    // the FXML grayscale checkbox
    @FXML
    private ImageView histogram;
    // the FXML area for showing the current frame
    @FXML
    private ImageView currentFrame;

    @FXML
    private TextField contrastInput;

    @FXML
    private TextField brightnessInput;

    private ScheduledExecutorService timer;
    private VideoCapture videoCapture = new VideoCapture();
    private boolean cameraActive = false;

    private Mat logo;

    static final String RESOURCE_FILE_BASE_NAME = "resources";

    @FXML
    private void startCamera() {
        if (!cameraActive) {
            videoCapture.open(0);
            if (videoCapture.isOpened()) {
                cameraActive = true;
                Runnable frameGrabber= new Runnable() {
                    @Override
                    public void run() {
                        Image image = grabFrame();
                        Platform.runLater(new Runnable() {
                            @Override
                            public void run() {
                                currentFrame.setImage(image);
                            }
                        });

                    }
                };
                timer = Executors.newSingleThreadScheduledExecutor();
                timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
                this.button.setText("Stop Camera");
            }
        } else {
            cameraActive = false;
            try {
                timer.shutdown();
                timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException exception) {
                System.err.println("Failed to stop the frame capture, release camera now");
            } finally {
                videoCapture.release();
                currentFrame.setImage(null);
                button.setText("Start Camera");
            }
        }
    }

    @FXML
    private void loadLogo() {
        if (logoCheckBox.isSelected()) {
            logo = Imgcodecs.imread(RESOURCE_FILE_BASE_NAME + "/Poli.png");
        }
    }

    private Image grabFrame() {
        Image image = null;
        Mat frame = new Mat();

        if (videoCapture.isOpened()) {
            videoCapture.read(frame);
            if (!frame.empty()) {

                if (logoCheckBox.isSelected() && logo != null) {
                    addLogoToFrame(frame);
                }

                if (grayscale.isSelected() && (contrastInput.getLength() > 0 && brightnessInput.getLength() > 0)) {
                    double contrast = Double.valueOf(contrastInput.getText());
                    int brightness = Integer.valueOf(brightnessInput.getText());
                    frame = changeBrightnessForFrame(frame, contrast, brightness);
                }

                generateHistogram(frame, grayscale.isSelected());
                image = mat2Image(frame);
            }
        }
        return image;
    }

    private void addLogoToFrame(Mat frame) {
        Rect rect = new Rect(frame.cols()-logo.cols(), frame.rows()-logo.rows(), logo.cols(), logo.rows());
        Mat matROI = frame.submat(rect);
        Core.addWeighted(matROI, 1.0, logo, 0.7, 0.0, matROI);
    }

    private void frameToGray(Mat frame) {
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
    }

    private Mat changeBrightnessForFrame(Mat frame, double contrast, int brightness) {
        Mat newFrame = Mat.zeros(frame.rows(), frame.cols(), frame.type());
        frame.convertTo(newFrame, -1, contrast, brightness);
        return newFrame;
    }

    private void generateHistogram(Mat frame, boolean gray) {
        List<Mat> splits = new ArrayList<>();
        Core.split(frame, splits);

        MatOfInt channels = new MatOfInt(0);
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat range = new MatOfFloat(0, 256);

        Mat hist_b = new Mat();
        Mat hist_g = new Mat();
        Mat hist_r = new Mat();

        Imgproc.calcHist(splits.subList(0, 1), channels, new Mat(), hist_b, histSize, range, false);
        if (!gray) {
            Imgproc.calcHist(splits.subList(1,2), channels, new Mat(), hist_g, histSize, range, false);
            Imgproc.calcHist(splits.subList(2,3), channels, new Mat(), hist_r, histSize, range, false);
        }

        int width = 150;
        int height = 150;
        int bin_w = (int)Math.round(width / histSize.get(0,0)[0]);
        Mat histImage = new Mat(width, height, CvType.CV_8UC3, new Scalar(0,0,0));

        Core.normalize(hist_b, hist_b, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
        if (!gray) {
            Core.normalize(hist_g, hist_g, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
            Core.normalize(hist_r, hist_r, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
        }

        for (int i = 1; i < histSize.get(0,0)[0]; i++) {
            Imgproc.line(histImage, new Point(bin_w * (i-1), height - Math.round(hist_b.get(i - 1, 0)[0])),
                    new Point(bin_w * i, height - Math.round(hist_b.get(i, 0)[0])), new Scalar(255, 0, 0), 2, 8, 0);
            if (!gray) {
                Imgproc.line(histImage, new Point(bin_w * (i-1), height - Math.round(hist_g.get(i - 1, 0)[0])),
                        new Point(bin_w * i, height - Math.round(hist_g.get(i, 0)[0])), new Scalar(0, 255, 0), 2, 8, 0);
                Imgproc.line(histImage, new Point(bin_w * (i-1), height - Math.round(hist_r.get(i - 1, 0)[0])),
                        new Point(bin_w * i, height - Math.round(hist_r.get(i, 0)[0])), new Scalar(0, 0, 255), 2, 8, 0);
            }
        }
        Image display = mat2Image(histImage);
        Platform.runLater(new Runnable() {
            @Override
            public void run() {
                histogram.setImage(display);
            }
        });
    }

    private Image mat2Image(Mat frame) {
        MatOfByte buff = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buff);
        return new Image(new ByteArrayInputStream(buff.toArray()));
    }
}
