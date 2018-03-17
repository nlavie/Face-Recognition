// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  In it, we will show how to do face recognition.  This example uses the
    pretrained dlib_face_recognition_resnet_model_v1 model which is freely available from
    the dlib web site.  This model has a 99.38% accuracy on the standard LFW face
    recognition benchmark, which is comparable to other state-of-the-art methods for face
    recognition as of February 2017.

    In this example, we will use dlib to do face clustering.  Included in the examples
    folder is an image, bald_guys.jpg, which contains a bunch of photos of action movie
    stars Vin Diesel, The Rock, Jason Statham, and Bruce Willis.   We will use dlib to
    automatically find their faces in the image and then to automatically determine how
    many people there are (4 in this case) as well as which faces belong to each person.

    Finally, this example uses a network with the loss_metric loss.  Therefore, if you want
    to learn how to train your own models, or to get a general introduction to this loss
    layer, you should read the dnn_metric_learning_ex.cpp and
    dnn_metric_learning_on_images_ex.cpp examples.
*/

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <math.h>       /* cos */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace dlib;
using namespace std;


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

std::vector<string> splitFiles(string pathes)
{
    std::vector<string> twoFolders;
    istringstream iss(pathes);
    string s;
    while ( getline( iss, s, '-' ) )
    {
      //cout<<s.c_str()<<endl;
      twoFolders.push_back(s.c_str());
    }

    return twoFolders;
}

int main(int argc, char** argv) try
{

     std::vector<string> pathes = splitFiles(argv[1]);
     string folderOrig =pathes[0];
     string folderList =pathes[1];





    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();


    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    ////image1/////////////////////////////////////////////////////////
    cv::String folderpathOriginal = folderOrig+"/*.jpg"; //"faces/test/original/*.jpg";
    std::vector<cv::String> filenamesOriginal;
    cv::glob(folderpathOriginal, filenamesOriginal);
    //cout<<"filenamesOriginal.size(); "<<filenamesOriginal.size()<<endl;

    matrix<rgb_pixel> imgOriginal;
    load_image(imgOriginal, filenamesOriginal[0]); //filenamesOriginal[0]
    // Display the raw image on the screen
    //image_window win(imgOriginal);

    std::vector<matrix<rgb_pixel>> facesOriginal;
    for (auto face : detector(imgOriginal))
    {
        auto shape = sp(imgOriginal, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(imgOriginal, get_face_chip_details(shape,150,0.25), face_chip);
        facesOriginal.push_back(move(face_chip));
        // Also put some boxes on the faces so we can see that the detector is finding
        // them.
        //win.add_overlay(face);
    }
    //cout<<"finish "<<endl;

    std::vector<matrix<float,0,1>> faceOrig_descriptors;
    if ( !(facesOriginal.size() == 0) )
    {
         faceOrig_descriptors = net(facesOriginal);

    }
    else
    {
        cout<<" cant find face in the original image"<<endl;
        return -1;
    }




    ////list/////////////////////////////////////////////////////////

    cv::String folderpathList = folderList+"/*.jpg";
    std::vector<cv::String> filenamesList;
    cv::glob(folderpathList, filenamesList);

    cout<<"list size origin "<<filenamesList.size()<<endl;
    std::vector<matrix<rgb_pixel>> facesCompares;
    for(int i=0; i <  filenamesList.size(); i++)
    {
        matrix<rgb_pixel> imgCompare;
        load_image(imgCompare, filenamesList[i]);
        facesCompares.push_back(imgCompare);

    }


    std::vector<matrix<rgb_pixel>> toNetfacesList;


    for(int i=0; i<facesCompares.size(); i++)
    {
        matrix<rgb_pixel> currentImgList =facesCompares[i] ;

        for (auto face : detector(currentImgList))
        {
            //cout<<" num of face "<<filenamesList[i]<<endl;
            auto shape = sp(currentImgList, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(currentImgList, get_face_chip_details(shape,150,0.25), face_chip);
            toNetfacesList.push_back(move(face_chip));
            // Also put some boxes on the faces so we can see that the detector is finding
            // them.
            //win2.add_overlay(face);
        }


    }

    std::vector<matrix<float,0,1>> faceList_descriptors;
    if ( !(toNetfacesList.size() == 0) )
    {
         faceList_descriptors = net(toNetfacesList);

    }
    else
    {
        cout<<" cant find faces in the list "<<endl;
        return -1;
    }






    //cout<<"bla bla  "<<endl;
    //matrix<float,0,1> descriptorOrig = faceOrig_descriptors[0];

    cout<<"the size of list = "<<faceList_descriptors.size()<<endl;
    for(int i=0; faceList_descriptors.size(); i++)
    {
        //cout<<"the size of list = "<<faceList_descriptors.size()<<endl;

        double l = length(faceOrig_descriptors[0]-faceList_descriptors[i]) ;
        //cout<<"i = "<<i<<endl;
        if (l< 0.6)
        {
            cout<<"the same person "<<l<<" ---> "<<filenamesList[i]<<endl;

        }
        else
        {
             cout<<"not!!! "<<l<<" ---> "<<filenamesList[i]<<endl;
        }

        if ( i == faceList_descriptors.size()-1 )
        {
            return 0;
        }


    }

    return 0;



}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

/// not in use
std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently.
    thread_local random_cropper cropper;
    cropper.set_chip_dims(150,150);
    cropper.set_randomly_flip(true);
    cropper.set_max_object_size(0.99999);
    cropper.set_background_crops_fraction(0);
    cropper.set_min_object_size(0.97);
    cropper.set_translate_amount(0.02);
    cropper.set_max_rotation_degrees(3);

    std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img),3);
    std::vector<matrix<rgb_pixel>> crops;

    matrix<rgb_pixel> temp;
    for (int i = 0; i < 100; ++i)
    {
        cropper(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(move(temp));
    }
    return crops;
}

// ----------------------------------------------------------------------------------------


