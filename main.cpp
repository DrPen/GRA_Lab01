/****************************************************************************\
* Vorlage fuer das Praktikum "Graphische Datenverarbeitung" WS 2018/19
* FB 03 der Hochschule Niedderrhein
* Regina Pohle-Froehlich
*
* Der Code basiert auf den c++-Beispielen der Bibliothek royale
\****************************************************************************/

#include <royale.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>


class MyListener : public royale::IDepthDataListener
{

public:

	void onNewData(const royale::DepthData *data)
	{
		// this callback function will be called for every new depth frame

		std::lock_guard<std::mutex> lock(flagMutex);
		zImage.create(cv::Size(data->width, data->height), CV_32FC1);
		grayImage.create(cv::Size(data->width, data->height), CV_32FC1);
		zImage = 0;
		grayImage = 0;
		int k = 0;
		for (int y = 0; y < zImage.rows; y++)
		{
			for (int x = 0; x < zImage.cols; x++)
			{
				auto curPoint = data->points.at(k);
				if (curPoint.depthConfidence > 0)
				{
					// if the point is valid
					zImage.at<float>(y, x) = curPoint.z;
					grayImage.at<float>(y, x) = curPoint.grayValue;
				}
				k++;
			}
		}

		cv::Mat temp = zImage.clone();
		undistort(temp, zImage, cameraMatrix, distortionCoefficients);
		temp = grayImage.clone();
		undistort(temp, grayImage, cameraMatrix, distortionCoefficients);
	}

	void setLensParameters(const royale::LensParameters &lensParameters)
	{
		// Construct the camera matrix
		// (fx   0    cx)
		// (0    fy   cy)
		// (0    0    1 )
		cameraMatrix = (cv::Mat1d(3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
			0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
			0, 0, 1);

		// Construct the distortion coefficients
		// k1 k2 p1 p2 k3
		distortionCoefficients = (cv::Mat1d(1, 5) << lensParameters.distortionRadial[0],
			lensParameters.distortionRadial[1],
			lensParameters.distortionTangential.first,
			lensParameters.distortionTangential.second,
			lensParameters.distortionRadial[2]);
	}

private:

	cv::Mat zImage, grayImage;
	cv::Mat cameraMatrix, distortionCoefficients;
	std::mutex flagMutex;
};

int main(int argc, char *argv[])
{
	MyListener listener;

	// this represents the main camera device object
	std::unique_ptr<royale::ICameraDevice> cameraDevice;

	// the camera manager will query for a connected camera
	{
		royale::CameraManager manager;

		// try to open the first connected camera
		royale::Vector<royale::String> camlist(manager.getConnectedCameraList());
		std::cout << "Detected " << camlist.size() << " camera(s)." << std::endl;

		if (!camlist.empty())
		{
			cameraDevice = manager.createCamera(camlist[0]);
		}
		else
		{
			std::cerr << "No suitable camera device detected." << std::endl
				<< "Please make sure that a supported camera is plugged in, all drivers are "
				<< "installed, and you have proper USB permission" << std::endl;
			return 1;
		}

		camlist.clear();

	}
	// the camera device is now available and CameraManager can be deallocated here

	if (cameraDevice == nullptr)
	{
		// no cameraDevice available
		if (argc > 1)
		{
			std::cerr << "Could not open " << argv[1] << std::endl;
			return 1;
		}
		else
		{
			std::cerr << "Cannot create the camera device" << std::endl;
			return 1;
		}
	}

	// call the initialize method before working with the camera device
	auto status = cameraDevice->initialize();
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Cannot initialize the camera device, error string : " << getErrorString(status) << std::endl;
		return 1;
	}

	// retrieve the lens parameters from Royale
	royale::LensParameters lensParameters;
	status = cameraDevice->getLensParameters(lensParameters);
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Can't read out the lens parameters" << std::endl;
		return 1;
	}

	listener.setLensParameters(lensParameters);

	// register a data listener
	if (cameraDevice->registerDataListener(&listener) != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error registering data listener" << std::endl;
		return 1;
	}

	// start capture mode
	if (cameraDevice->startCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error starting the capturing" << std::endl;
		return 1;
	}

	// stop capture mode
	if (cameraDevice->stopCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error stopping the capturing" << std::endl;
		return 1;
	}

	return 0;
}
