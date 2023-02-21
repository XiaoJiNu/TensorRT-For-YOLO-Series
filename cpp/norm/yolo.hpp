#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.2

using namespace nvinfer1;
static Logger gLogger;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_yolo_proposals(float* feat_blob, int output_size, float prob_threshold, std::vector<Object>& objects)
{
    const int num_class = 80;
    auto dets = output_size / (num_class + 5);
    for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
    {
        const int basic_pos = boxs_idx *(num_class + 5);
        float x_center = feat_blob[basic_pos+0];
        float y_center = feat_blob[basic_pos+1];
        float w = feat_blob[basic_pos+2];
        float h = feat_blob[basic_pos+3];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float box_objectness = feat_blob[basic_pos+4];
        // std::cout<<*feat_blob<<std::endl;
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
    }

}

static void decode_outputs(float* prob, int output_size, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        generate_yolo_proposals(prob, output_size, BBOX_CONF_THRESH, proposals);
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    cv::imwrite("det_res.jpg", image);
    fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
}

static void show_img(cv::Mat pr_img) {
    cv::namedWindow("pr_img");
    cv::imshow("pr_img", pr_img);
    cv::waitKey(0);
}

class YOLO
{
//    public:
//        YOLO(std::string engine_file_path);
//        virtual ~YOLO();
//        void detect_img(std::string image_path);
//        void detect_video(std::string video_path);
//        cv::Mat static_resize(cv::Mat& img);
//        float* blobFromImage(cv::Mat& img);
//        void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape);
//
//    private:
//        static const int INPUT_W = 640;
//        static const int INPUT_H = 640;
//        const char* INPUT_BLOB_NAME = "image_arrays";
//        const char* OUTPUT_BLOB_NAME = "outputs";
//        float* prob;
//        int output_size = 1;
//        ICudaEngine* engine;
//        IRuntime* runtime;
//        IExecutionContext* context;

public:
    YOLO(std::string engine_file_path);
    virtual ~YOLO();  // 虚析构函数的作用？？没有虚函数的类是不是不应该用虚析构函数？
    void detect_img(std::string image_path);
    void detect_video(std::string video_path);
    cv::Mat static_resize(cv::Mat& img);  // 引用传参数？？和指针传参的区别？
    float* blobFromImage(cv::Mat& img);    // 归一化操作？还做了rbg -> bgr?
    // IExecutionContext ?? cv::Size ??
    void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape);

private:
    // 静态变量的作用？？ const不同位置不同含义？
    static const int INPUT_W = 640;
    static const int INPUT_H = 640;
    const char* INPUT_BLOB_NAME = "image_arrarys";  // 指针INPUT_BLOB_NAME指向的字符是不可变的(即常量字符)
    const char* OUTPUT_BLOB_NAME = "outputs";       // 指针OUTPUT_BLOB_NAME指向的字符是不可变的
    float* prob;
    int output_size = 1;
    IRuntime* runtime;   // IRuntime ??  运行实例
    ICudaEngine* engine; // ICudaEngine ??  engine
    IExecutionContext* context; //IExecutionContext??  上下文，管理中间激活的额外状态

};

YOLO::YOLO(std::string engine_file_path)
{
    // 读入模型，保存在字符串中 -> 然后生成runtime -> 反序列化模型得到engine -> 生成执行上下文
    // -> 开辟保存输出tensor的float数组空间 -> 删除保存模型的字符串数组
    size_t size{0};
    // 字符串指针。cout << trtModelStream为输出整个字符串，cout << *trtModelStream 输出该字符串得第一个字符
    // 参考 https://blog.csdn.net/Kallou/article/details/123239999#:~:text=C%2B%2B%E5%A4%84%E7%90%86%E5%AD%97%E7%AC%A6%E4%B8%B2%E6%9C%89%E4%B8%A4%E7%A7%8D%E6%96%B9%E5%BC%8F%EF%BC%8C%E5%8D%B3%EF%BC%9A%20%E6%8C%87%E9%92%88%E6%96%B9%E5%BC%8F%E5%92%8C%E6%95%B0%E7%BB%84%E6%96%B9%E5%BC%8F%20%E6%95%B0%E7%BB%84%E6%96%B9%E5%BC%8F%EF%BC%9Achar%20a%20%5B%5D%20%3D,%22HelloWorld%22%3B%20%E6%8C%87%E9%92%88%E6%96%B9%E5%BC%8F%EF%BC%9Aconst%20char%2A%20s%20%3D%20%22HelloWorld%22%3B%20const%E5%8F%AF%E4%BB%A5%E5%BF%BD%E7%95%A5
    char *trtModelStream{nullptr};

    // 创建输入流并打开模型，然后将模型输入流存入字符指针中
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);  // 从输入流的末尾，向前移动0个位置。即流移动到末尾位置
        size = file.tellg();      // 查询流当前位置，即末尾所在位置的字节数，也即读入的模型字节大小
        file.seekg(0, file.beg);  // 从输入流的开头位置，向前流移皮革位置。即流动到文件开头位置。
        trtModelStream = new char[size];  // 开辟模型大小的空间，用于保存模型
        assert(trtModelStream);
        // 读入file流中缓存的模型，将模型存放入trtModelStream字符数组指针中，读入的大小为模型大小字节
        file.read(trtModelStream, size);
        file.close();
    }
    std::cout << "engine init finished" << std::endl;

    // 创建runtime实例
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    // 反序列化模型，得到engine实例。这里没有用*trtModelStream，因为*trtModelStream表示取字符串第一个字符
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    // 创建上下文
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;  // 反序列化模型后，可以删除保存模型的字符串数组
    // TODO 获取输出tensor的维度，这里是在生成网络时的bindings的维度吗，没有看到生成trt模型中有bindings
    // 分配输出tensor的空间
    auto out_dims = engine->getBindingDimensions(1);
    for(int j=0;j<out_dims.nbDims;j++) {
        // 这里得到输出tensor 维度1x8400x85的乘积，用于分配输出空间大小
        this->output_size *= out_dims.d[j];
    }
    this->prob = new float[this->output_size];  // 开辟用于保存输出tensor的空间
}

YOLO::~YOLO()
{
    std::cout<<"yolo destroy"<<std::endl;
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    
}

void YOLO::detect_img(std::string image_path)
{
//    // 1. 读入图片并resize
//    cv::Mat img = cv::imread(image_path);
//    int img_w = img.cols;
//    int img_h = img.rows;
//    // 623x618 -> 640x640，img为引用传参
//    cv::Mat pr_img = this->static_resize(img);  // ??
////    cv::namedWindow("pr_img");
////    cv::imshow("pr_img", pr_img);
////    cv::waitKey(0);
//    std::cout << "blob image" << std::endl;
//
//    // 2. 归一化 ？？
//    float* blob;
//    blob = blobFromImage(pr_img);  // 只是归一化操作？？, pr_img为引用传参
//    float scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));
//
//    // run inference
//    // 3. 推理图片
//    auto start = std::chrono::system_clock::now();
//    // 为什么context可以直接被调用，pro用this->prob调用？？
//    // 为什么context为引用传参？？ input,output用指针传参？？
//    // ** doInference中，模型输出结果保存在指针prob中，所以后续decode时直接调用prob即可，没有直接返回模型推理结果
//    doInference(*context, blob, this->prob, output_size, pr_img.size());
//    auto end = std::chrono::system_clock::now();
//    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//
//    // 4. decode 模型输出的所有结果框
//    std::vector<Object> objects;
//    decode_outputs(this->prob, this->output_size, objects, scale, img_w, img_h);
//    draw_objects(img, objects, image_path);
//    delete blob;

    // 1. 读入图片并resize
    cv::Mat img = cv::imread(image_path);
    int img_w = img.rows;
    int img_h = img.cols;
    // img的维度变化是怎样？？
    cv::Mat pr_img = static_resize(img);
    std::cout << "blob image" << std::endl;

    // 2. 归一化 ？？
    float* blob;
    blob = blobFromImage(pr_img);
    // scale用于干什么
    float scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));

    // 3. 推理图片
    // chrono是一个time library
    auto start = std::chrono::system_clock::now();
    doInference(*context, blob, this->prob, output_size, pr_img.size());
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl ;
    int temp = 0;

    // 4. decode 模型输出的所有结果框
    std::vector<Object> objects;
    decode_outputs(this->prob, this->output_size, objects, scale, img_w, img_h);
}

cv::Mat YOLO::static_resize(cv::Mat& img) {
//    float r = std::min(this->INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
//    int unpad_w = r * img.cols;
//    int unpad_h = r * img.rows;
//    // re是按照图像比例resize到640尺度，其中宽高对应缩放比例小的那边刚好到640,另一边按照比例r进行resize。此时图像还没有填充到640×640
//    cv::Mat re(unpad_h, unpad_w, CV_8UC3);  // re: hxw = 640x634
//    cv::resize(img, re, re.size());
//    // 生成一个640*640的灰度图片，用于将得到得按照比例缩放后的图片放进这个灰度图片
//    cv::Mat out(this->INPUT_W, this->INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
//    // src.copyTo(dst)和src.clone(dst)效果相同，都是深拷贝。只是当src没有分配内存时候，copyTo为src分配内存，如果有，则不分配内存。
//    // 但是clone一定是会为src重新分配内存。clone是先重新分配内存，再调用copyTo
//    // https://blog.csdn.net/yangshengwei230612/article/details/102758136?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-102758136-blog-70154719.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-102758136-blog-70154719.pc_relevant_aa&utm_relevant_index=5
//    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
//
//    // 测试改变copyTo得到的dst会不会改变src
//    // 经过下面的代码确认，src.copyTo(dst)为深拷贝，只是比clone快
//    for (size_t i = 0; i < 3; ++i) {
//        for (size_t h = 0; h < out.rows; ++h) {
//            for (size_t w = 0; w < out.cols; ++w) {
//                out.at<cv::Vec3b>(h, w)[i] = 0;
//            }
//        }
//    }
//    show_img(out);
//
//    return out;

    // 1. 将图片得长边缩放为640,短边按照比例缩放，保持长宽比例不变
    // 得到所放比例小得那条边的缩放比例为r.这条边按照r缩放为640，而另一条边按照r所放小于等于640
    float r = std::min(this->INPUT_H / img.rows, this->INPUT_W / img.cols);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size()); // re.size() ??

    // 2. 生成一个640*640的Mat数组out，用于存放缩放后得图片
    cv::Mat out(this->INPUT_H, this->INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    // 3. 将图片放到out数组中，这里是将re数组复制到out的ROI区域
    // cv::Rect() ?? out(cv::Rect(0, 0, re.cols, re.rows))是提取out数组得左上角坐标(0,0)，宽高为w,h的区域
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

}

float* YOLO::blobFromImage(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return blob;
}

void YOLO::doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    // 获取关联的引擎.这里engine为引用别名，并且为常量，不能更改
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // 实现功能：创建一个指针，它指向device上的输入、输出缓冲区，并将传递给engine
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // Engine 需要的缓冲区数量必须和Engine::getNbBindings()得到得数量一致
    assert(engine.getNbBindings() == 2);
    void* buffers[2];  // 用于存放输入输出缓冲区

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // 为了绑定缓冲区，我们需要知道输入、输出tensor的名字
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // 必须确保索引数量要小于IEngine::getNbBindings()得到的值
    // INPUT_BLOB_NAME什么时候绑定到模型的？？
    // getBindingIndex实现什么功能??
    // 答：Retrieve the binding index for a named tensor.给定一个tensor名字获取它在binding缓冲区的索引
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);  // 得到输入tensor在binding缓冲区的索引
    // getBindingDataType(inputIndex)，确认输入tensor的数据类型是否为32位浮点数
    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    // 在gpu上分配用于存储输入、输出tensor的缓冲区，并让buffers的两个指针变量指向它们
    // &buffers[inputIndex]中的&实现什么功能？？
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    // 创建异步流。流实现的功能是什么？？
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 将输入tensor复制到gpu上的缓冲区，这里为什么传入流？？
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    // 推理模型
    context.enqueue(1, buffers, stream, nullptr);
    // 将模型输出tensor从gpu复制到cpu缓冲区
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream); // 等待流任务结束

    // Release stream and buffers，释放流和输入输出缓冲区
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}