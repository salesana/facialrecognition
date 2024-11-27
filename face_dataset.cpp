#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open the camera" << endl;
        return -1;
    }

    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
        cerr << "Error: Could not load Haar Cascade file" << endl;
        return -1;
    }

    string dataset_path = "./face_dataset/";
    if (!fs::exists(dataset_path)) {
        fs::create_directory(dataset_path);
        cout << "Created directory: " << dataset_path << endl;
    }

    // Get the person's name
    cout << "Enter the name of the person: ";
    string person_name;
    cin >> person_name;

    vector<vector<double>> face_data;
    int sample_count = 0;
    const int MAX_SAMPLES = 25;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Could not capture frame" << endl;
            continue;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);

        for (const auto& face : faces) {
            int x = face.x, y = face.y, w = face.width, h = face.height;

            // Extract face ROI
            int offset = 5;
            Rect roi(
                max(0, x - offset),
                max(0, y - offset),
                min(face.width + 2 * offset, frame.cols - x),
                min(face.height + 2 * offset, frame.rows - y)
            );

            Mat face_section = frame(roi);
            resize(face_section, face_section, Size(100, 100));

            // Flatten face_section
            vector<double> face_vector;
            face_vector.reserve(100 * 100);
            for (int i = 0; i < face_section.rows; ++i) {
                for (int j = 0; j < face_section.cols; ++j) {
                    face_vector.push_back(face_section.at<uchar>(i, j));
                }
            }

            // Add face data to dataset if sample count < MAX_SAMPLES
            if (sample_count < MAX_SAMPLES) {
                face_data.push_back(face_vector);
                sample_count++;
                cout << "Collected sample " << sample_count << " of " << MAX_SAMPLES << endl;

                // Draw bounding box and label
                rectangle(frame, face, Scalar(0, 255, 0), 2);
                putText(frame, "Sample collected", Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            } else {
                putText(frame, "Sample limit reached", Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }
        }

        // Display the frame
        imshow("Face Collection", frame);

        // Exit on pressing 'q'
        if (waitKey(1) == 'q' || sample_count >= MAX_SAMPLES) break;
    }

    // Save collected face samples
    string output_file = dataset_path + person_name + ".bin";
    ofstream file(output_file, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not save face data" << endl;
        return -1;
    }

    for (const auto& face : face_data) {
        file.write((char*)face.data(), face.size() * sizeof(double));
    }
    file.close();

    cout << "Collected " << sample_count << " samples. Dataset saved at: " << output_file << endl;

    cap.release();
    destroyAllWindows();

    return 0;
}
