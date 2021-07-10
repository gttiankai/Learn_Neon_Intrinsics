#include <jni.h>
#include <string>
#include <chrono>
#include <arm_neon.h>

short *generateRamp(short start_value, short len) {
    auto *ramp = new short[len];
    for (int i = 0; i < len; ++i) {
        ramp[i] = start_value + i;
    }
    return ramp;
}

double msElapsedTime(std::chrono::system_clock::time_point start) {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::chrono::system_clock::time_point now() {
    return std::chrono::system_clock::now();
}

int DotProduct(const short *vector1, const short *vector2, short len) {
    int result = 0;

    for (short i = 0; i < len; ++i) {
        result += vector1[i] * vector2[i];
    }
    return result;
}


int DotProductNeon(const short *vector1, const short *vector2, short len) {

    const short transfer_size = 4;
    short segments = len / transfer_size;
    // 4 element vector of zeros
    int32x4_t partial_sums_neon = vdupq_n_s32(0);

    // main loop (note the loop index goes through segments
    for (int i = 0; i < segments; ++i) {
        // load vector to element to registers
        short offset = i * transfer_size;
        int16x4_t vector1_neon = vld1_s16(vector1 + offset);
        int16x4_t vector2_neon = vld1_s16(vector2 + offset);

        // multiply and accumulate: partialSumsNeon += vector1_neon * vector2_neon;
        partial_sums_neon = vmlal_s16(partial_sums_neon, vector1_neon, vector2_neon);
    }

    // store partial sums
    int partial_sums[transfer_size];
    vst1q_s32(partial_sums, partial_sums_neon);

    // sum up partial sums
    int result = 0;
    for (short i = 0; i < transfer_size; i++) {
        result += partial_sums[i];
    }
    return result;

}


extern "C" JNIEXPORT jstring JNICALL
Java_com_learn_neon_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    const int ramp_length = 1024;
    const int trials = 10000;
    auto ramp1 = generateRamp(0, ramp_length);
    auto ramp2 = generateRamp(100, ramp_length);

    int last_result = 0;
    auto start = now();
    for (int i = 0; i < trials; ++i) {
        last_result = DotProduct(ramp1, ramp2, ramp_length);
    }
    auto elapsedTime = msElapsedTime(start);

    int last_result_neon = 0;
    start = now();
    for (int i = 0; i < trials; ++i) {
        last_result_neon = DotProductNeon(ramp1, ramp2, ramp_length);
    }
    auto elapsedTimeNeon = msElapsedTime(start);
    delete[]ramp1;
    delete[]ramp2;
    std::string resultsString =
            "----==== NO NEON ====----\nResult: " + std::to_string(last_result)
            + "\nElapsed time: " + std::to_string((int) elapsedTime) + " ms"
            + "\n\n----==== NEON ====----\n"
            + "Result: " + std::to_string(last_result_neon)
            + "\nElapsed time: " + std::to_string((int) elapsedTimeNeon) + " ms";
    return env->NewStringUTF(resultsString.c_str());
}

