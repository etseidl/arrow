// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/util/compression_internal.h"

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <snappy.h>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"

#include <nvcomp.h>
#include <nvcomp/snappy.h>

using std::size_t;

namespace arrow {
namespace util {
namespace internal {

namespace {

// ----------------------------------------------------------------------
// Snappy implementation

class SnappyCodec : public Codec {
    private:
        cudaStream_t stream_;
        size_t * gpu_in_sizes_;
        void ** gpu_in_ptrs_;
        void * gpu_in_data_;
        size_t in_data_size_;
        size_t * gpu_out_sizes_;
        void ** gpu_out_ptrs_;
        void * gpu_out_data_;
        size_t out_data_size_;
        void * gpu_temp_buf_;
        size_t gpu_temp_buf_size_;
        
        void checkErr(cudaError_t stat) {
            if (stat != cudaSuccess) {            \
                std::cerr << "cuda call failed" << std::endl;
                std::cerr << "status = " << stat << std::endl;
                exit(-1);
            }
        }
        void checkErr(nvcompError_t stat) {
            if (stat != nvcompSuccess) {            \
                std::cerr << "nvcomp call failed" << std::endl;
                std::cerr << "status = " << stat << std::endl;
                exit(-1);
            }
        }
        
 public:
    SnappyCodec()
        : in_data_size_(0), out_data_size_(0), gpu_temp_buf_size_(0)
    {
        //std::cout << "create stream" << std::endl;
        cudaStreamCreate(&stream_);
        //std::cout << "init malloc" << std::endl;
        checkErr(cudaMalloc((void**)(&gpu_in_sizes_), sizeof(size_t)));
        checkErr(cudaMalloc((void**)(&gpu_out_sizes_), sizeof(size_t)));
        checkErr(cudaMalloc((void**)(&gpu_in_ptrs_), sizeof(void*)));
        checkErr(cudaMalloc((void**)(&gpu_out_ptrs_), sizeof(void*)));
        //std::cout << "ok" << std::endl;
    }
        
  Result<int64_t> Decompress(int64_t input_len, const uint8_t* input,
                             int64_t output_buffer_len, uint8_t* output_buffer) override {
    size_t decompressed_size;
    if (!snappy::GetUncompressedLength(reinterpret_cast<const char*>(input),
                                       static_cast<size_t>(input_len),
                                       &decompressed_size)) {
      return Status::IOError("Corrupt snappy compressed data.");
    }
    if (output_buffer_len < static_cast<int64_t>(decompressed_size)) {
      return Status::Invalid("Output buffer size (", output_buffer_len, ") must be ",
                             decompressed_size, " or larger.");
    }
    if (!snappy::RawUncompress(reinterpret_cast<const char*>(input),
                               static_cast<size_t>(input_len),
                               reinterpret_cast<char*>(output_buffer))) {
      return Status::IOError("Corrupt snappy compressed data.");
    }
    return static_cast<int64_t>(decompressed_size);
  }

  int64_t MaxCompressedLen(int64_t input_len,
                           const uint8_t* ARROW_ARG_UNUSED(input)) override {
    DCHECK_GE(input_len, 0);
    return snappy::MaxCompressedLength(static_cast<size_t>(input_len));
  }

  Result<int64_t> Compress(int64_t input_len, const uint8_t* input,
                           int64_t ARROW_ARG_UNUSED(output_buffer_len),
                           uint8_t* output_buffer) override {
    size_t output_size;
#if 1
    void * inp[1];
    void * outp[1];
    size_t inlen[1] = {(size_t)input_len};

    // copy input size, input data to gpu buffers
    //std::cout << "in comp: ck in" << std::endl;
    checkErr(cudaMemcpy(gpu_in_sizes_, inlen, sizeof(size_t), cudaMemcpyHostToDevice));
    if ((size_t)input_len > in_data_size_) {
        //std::cout << "realloc input " << input_len << std::endl;
        if (in_data_size_)
            cudaFree(gpu_in_data_);
        checkErr(cudaMalloc((void**)(&gpu_in_data_), input_len));
        in_data_size_ = input_len;
        inp[0] = gpu_in_data_;
        cudaMemcpy((void**)(gpu_in_ptrs_), (void**)(inp), sizeof(void*),
                   cudaMemcpyHostToDevice);
    }
    checkErr(cudaMemcpy(gpu_in_data_, input, input_len, cudaMemcpyHostToDevice));

    // ensure temp buf is big enough
    //std::cout << "in comp: ck temp" << std::endl;
    size_t temp_bytes;
    checkErr(nvcompBatchedSnappyCompressGetTempSize(1, input_len, &temp_bytes));
    //std::cout << "input_len " << input_len << std::endl;
    //std::cout << "temp_bytes " << temp_bytes << std::endl;
    //std::cout << "gpu_temp_bytes " << gpu_temp_buf_size_ << std::endl;
    temp_bytes = 1; // always returns 0 for snappy
    
    if (gpu_temp_buf_size_ < temp_bytes) {
        //std::cout << "realloc temp " << temp_bytes << std::endl;
        if (gpu_temp_buf_size_)
            cudaFree(gpu_temp_buf_);
        cudaMalloc((void**)(&gpu_temp_buf_), temp_bytes);
        gpu_temp_buf_size_ = temp_bytes;
    }
    
    // ensure output buf is big enough
    //std::cout << "in comp: ck out" << std::endl;
    size_t comp_out_bytes;
    checkErr(nvcompBatchedSnappyCompressGetOutputSize(input_len, &comp_out_bytes));
    //std::cout << "out bytes " << comp_out_bytes << std::endl;
    if (comp_out_bytes > out_data_size_) {
        if (out_data_size_)
            cudaFree(gpu_out_data_);
        checkErr(cudaMalloc((void**)(&gpu_out_data_), comp_out_bytes));
        out_data_size_ = comp_out_bytes;
        outp[0] = gpu_out_data_;
        cudaMemcpy((void**)(gpu_out_ptrs_), (void**)(outp), sizeof(void*),
                   cudaMemcpyHostToDevice);
    }

    //std::cout << "in comp: do comp" << std::endl;
    checkErr(nvcompBatchedSnappyCompressAsync(
        (const void* const*)gpu_in_ptrs_,
        gpu_in_sizes_,
        1,
        gpu_temp_buf_,
        gpu_temp_buf_size_,
        gpu_out_ptrs_,
        gpu_out_sizes_,
        stream_));

    //std::cout << "in comp: do sync" << std::endl;
    checkErr(cudaStreamSynchronize(stream_));
    
    //std::cout << "in comp: get output" << std::endl;
    cudaMemcpy(&output_size, gpu_out_sizes_, sizeof(size_t), cudaMemcpyDeviceToHost);
    //std::cout << "comp size " << output_size << std::endl;
    cudaMemcpy(output_buffer, gpu_out_data_, output_size, cudaMemcpyDeviceToHost);
    //std::cout << "in comp: done" << std::endl;
#else
    snappy::RawCompress(reinterpret_cast<const char*>(input),
                        static_cast<size_t>(input_len),
                        reinterpret_cast<char*>(output_buffer), &output_size);
#endif
    return static_cast<int64_t>(output_size);
  }

  Result<std::shared_ptr<Compressor>> MakeCompressor() override {
    return Status::NotImplemented("Streaming compression unsupported with Snappy");
  }

  Result<std::shared_ptr<Decompressor>> MakeDecompressor() override {
    return Status::NotImplemented("Streaming decompression unsupported with Snappy");
  }

  Compression::type compression_type() const override { return Compression::SNAPPY; }
  int minimum_compression_level() const override { return kUseDefaultCompressionLevel; }
  int maximum_compression_level() const override { return kUseDefaultCompressionLevel; }
  int default_compression_level() const override { return kUseDefaultCompressionLevel; }
};

}  // namespace

std::unique_ptr<Codec> MakeSnappyCodec() {
  return std::unique_ptr<Codec>(new SnappyCodec());
}

}  // namespace internal
}  // namespace util
}  // namespace arrow
