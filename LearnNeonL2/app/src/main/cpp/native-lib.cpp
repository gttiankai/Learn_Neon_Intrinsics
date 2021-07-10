#include <jni.h>
#include <string>
#include <arm_neon.h>
#include <chrono>
#include <math.h>


extern "C" JNIEXPORT jstring JNICALL
Java_com_learn_learnneonl2_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

#define SIGNAL_LENGTH 10240
#define SIGNAL_AMPLITUDE 100
#define NOISE_AMPLITUDE 25
#define THRESHOLD 50
#define KERNEL_LENGTH 16
int8_t input_signal[SIGNAL_LENGTH];
int8_t input_signal_truncate[SIGNAL_LENGTH];
int8_t input_signal_convolution[SIGNAL_LENGTH];
double processingTime;
#define M_PI 3.1415926
// Kernel
int8_t kernel[KERNEL_LENGTH] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

double usElapsedTime(std::chrono::system_clock::time_point start) {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}


std::chrono::system_clock::time_point now() {
    return std::chrono::system_clock::now();
}

void generateSignal() {
    auto phase_step = 2 * M_PI / SIGNAL_LENGTH;
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        auto phase = i * phase_step;
        auto noise = rand() % NOISE_AMPLITUDE;
        input_signal[i] = static_cast<int8_t>(SIGNAL_AMPLITUDE * sin(phase) + noise);
    }
}

void Truncate() {
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        input_signal_truncate[i] = std::min(input_signal[i], (int8_t) THRESHOLD);
    }
}

void TruncateNeon() {
    int8x16_t threshold_neon = vdupq_n_s8(THRESHOLD);
    short slice = 16;
    int slice_count = SIGNAL_LENGTH / slice;
    for (int i = 0; i < slice_count; ++i) {
        int8x16_t input_neon = vld1q_s8(input_signal + i * 16);

        // truncate
        uint8x16_t partial_result = vminq_s8(input_neon, threshold_neon);
        // store result in the output buffer
        vst1q_s8(input_signal_truncate + i * 16, partial_result);
    }
}


int GetSum(const int8_t *input, int length) {
    int sum = 0;
    for (int i = 0; i < length; ++i) {
        sum += input[i];
    }
    return sum;
}


void Convolution() {
    auto offset = -KERNEL_LENGTH / 2;
    auto kernel_sum = GetSum(kernel, KERNEL_LENGTH);
    // Calculate convolution
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        int conv_sum = 0;
        for (int j = 0; j < KERNEL_LENGTH; ++j) {
            conv_sum += kernel[j] * input_signal[i + offset + j];
        }
        input_signal_convolution[i] = (uint8_t) (conv_sum / kernel_sum);
    }
}

void ConvolutionNeon() {
    auto offset = -KERNEL_LENGTH / 2;
    auto kernel_sum = GetSum(kernel, KERNEL_LENGTH);
    int8x16_t kernel_neon = vld1q_s8(kernel);
    int8_t *mul_result = new int8_t[16];
    for (int i = 0; i < SIGNAL_LENGTH; i += 16) {
        int8x16_t input_neon = vld1q_s8(input_signal + i + offset);
        int8x16_t mul_result_neon = vmulq_s8(input_neon, kernel_neon);
        vst1q_s8(mul_result, mul_result_neon);
        auto conv_sum = GetSum(mul_result, 16);
        input_signal_convolution[i] = (uint8_t) (conv_sum / kernel_sum);
    }
}


jbyteArray nativeBufferToByteArray(JNIEnv *env,
                                   int8_t *buffer, int length) {
    auto byteArray = env->NewByteArray(length);

    env->SetByteArrayRegion(byteArray, 0, length, buffer);

    return byteArray;
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_learn_learnneonl2_MainActivity_generateSignal(JNIEnv *env, jobject thiz) {
    generateSignal();

    return nativeBufferToByteArray(env, input_signal, SIGNAL_LENGTH);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_learn_learnneonl2_MainActivity_truncate(JNIEnv *env, jobject thiz, jboolean use_neon) {
    auto start = now();
    if (use_neon) {
        TruncateNeon();
    } else {
        Truncate();
    }
    processingTime = usElapsedTime(start);
    return nativeBufferToByteArray(env, input_signal_truncate, SIGNAL_LENGTH);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_learn_learnneonl2_MainActivity_convolution(JNIEnv *env, jobject thiz, jboolean use_neon) {
    auto start = now();
    if (use_neon) {
        ConvolutionNeon();
    } else {
        Convolution();
    }

    processingTime = usElapsedTime(start);
    return nativeBufferToByteArray(env, input_signal_convolution, SIGNAL_LENGTH);
}

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_learn_learnneonl2_MainActivity_getProcessingTime(JNIEnv *env, jobject thiz) {
    return processingTime;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_learn_learnneonl2_MainActivity_getSignalLength(JNIEnv *env, jobject thiz) {
    return SIGNAL_LENGTH;
}