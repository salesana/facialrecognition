#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }

    CascadeClassifier face_cascade;
    if (!face_cascade.load(cv::samples::findFile("haarcascade_frontalface_alt.xml"))) {
        cerr << "Error: Could not load Haar Cascade file" << endl;
        return -1;
    }

    string dataset_path = "./face_dataset/";
    if (!fs::exists(dataset_path)) {
        fs::create_directory(dataset_path);
        cout << "Created directory: " << dataset_path << endl;
    }

    cout << "Enter the name of the person: ";
    string file_name;
    cin >> file_name;

    int skip = 0;
    vector<Mat> face_data;

    while (true) {
        Mat frame, gray_frame;
        cap >> frame;

        if (frame.empty()) {
            cerr << "Failed to capture frame" << endl;
            continue;
        }

        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray_frame, faces, 1.3, 5);

        if (faces.empty()) continue;

        // Sort faces by area (largest first)
        sort(faces.begin(), faces.end(), [](const Rect &a, const Rect &b) {
            return a.area() > b.area();
        });

        skip++;

        for (size_t i = 0; i < 1 && i < faces.size(); ++i) {
            Rect face = faces[i];
            int offset = 5;

            Rect expanded_face(
                max(face.x - offset, 0),
                max(face.y - offset, 0),
                min(face.width + 2 * offset, frame.cols - face.x + offset),
                min(face.height + 2 * offset, frame.rows - face.y + offset)
            );

            Mat face_offset = frame(expanded_face);
            Mat face_selection;
            resize(face_offset, face_selection, Size(100, 100));

            if (skip % 10 == 0) {
                face_data.push_back(face_selection.clone());
                cout << "Collected " << face_data.size() << " face samples" << endl;
            }

            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        imshow("faces", frame);

        if (waitKey(1) == 'q') break;
    }

    // Save face data to a binary file
    string output_file = dataset_path + file_name + ".bin";
    ofstream file(output_file, ios::binary);

    if (!file.is_open()) {
        cerr << "Error: Could not save face data" << endl;
        return -1;
    }

    for (const auto &face : face_data) {
        file.write((char *)face.data, face.total() * face.elemSize());
    }
    file.close();

    cout << "Dataset saved at: " << output_file << endl;

    cap.release();
    destroyAllWindows();
    return 0;
}
