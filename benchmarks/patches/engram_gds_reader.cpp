// Experimental cuFile page reader. The caller owns the CUDA buffer lifetime.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime_api.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static void check(CUfileError_t status, const char* operation) {
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuFile error " +
                                 std::to_string(status.err));
    }
}

class PageReader {
    py::object owner_;
    int fd_ = -1;
    int device_;
    void* buffer_;
    size_t capacity_;
    CUfileHandle_t file_ = nullptr;
    CUfileBatchHandle_t batch_ = nullptr;
    bool registered_ = false;

    void cleanup() noexcept {
        cudaSetDevice(device_);
        if (batch_) { cuFileBatchIODestroy(batch_); batch_ = nullptr; }
        if (registered_) { cuFileBufDeregister(buffer_); registered_ = false; }
        if (file_) { cuFileHandleDeregister(file_); file_ = nullptr; }
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
    }

public:
    PageReader(const std::string& path, uintptr_t pointer, size_t capacity, int device, py::object owner)
        : owner_(std::move(owner)), device_(device), buffer_(reinterpret_cast<void*>(pointer)), capacity_(capacity) {
        if (capacity == 0 || pointer % 4096 != 0) {
            throw std::runtime_error("GDS requires a nonempty page-aligned CUDA buffer");
        }
        if (cudaSetDevice(device_) != cudaSuccess) {
            throw std::runtime_error("Cannot select CUDA device for GDS");
        }
        try {
            check(cuFileDriverOpen(), "cuFileDriverOpen");
            fd_ = open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd_ < 0) throw std::runtime_error("Cannot open GDS file with O_DIRECT");
            CUfileDescr_t descriptor{};
            descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            descriptor.handle.fd = fd_;
            check(cuFileHandleRegister(&file_, &descriptor), "cuFileHandleRegister");
            check(cuFileBufRegister(buffer_, capacity_ * 4096, 0), "cuFileBufRegister");
            registered_ = true;
            check(cuFileBatchIOSetUp(&batch_, 128), "cuFileBatchIOSetUp");
        } catch (...) { cleanup(); throw; }
    }
    PageReader(const PageReader&) = delete;
    ~PageReader() { cleanup(); }

    size_t read(const std::vector<int64_t>& pages) {
        if (pages.size() > capacity_) throw std::runtime_error("GDS page buffer capacity exceeded");
        for (auto page : pages) {
            if (page < 0 || page > INT64_MAX / 4096) {
                throw std::runtime_error("Invalid GDS file page");
            }
        }
        if (cudaSetDevice(device_) != cudaSuccess) throw std::runtime_error("Cannot select CUDA device");
        py::gil_scoped_release release;
        for (size_t begin = 0; begin < pages.size(); begin += 128) {
            unsigned count = std::min<size_t>(128, pages.size() - begin);
            std::vector<CUfileIOParams_t> requests(count);
            for (unsigned i = 0; i < count; ++i) {
                auto& request = requests[i];
                request.mode = CUFILE_BATCH;
                request.opcode = CUFILE_READ;
                request.fh = file_;
                request.u.batch.devPtr_base = buffer_;
                request.u.batch.devPtr_offset = (begin + i) * 4096;
                request.u.batch.file_offset = pages[begin + i] * 4096;
                request.u.batch.size = 4096;
            }
            check(cuFileBatchIOSubmit(batch_, count, requests.data(), 0), "cuFileBatchIOSubmit");
            unsigned completed = 0;
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            bool short_read = false;
            try {
                while (completed < count) {
                    CUfileIOEvents_t events[128]{};
                    unsigned returned = count - completed;
                    timespec timeout{1, 0};
                    check(cuFileBatchIOGetStatus(batch_, 1, &returned, events, &timeout),
                          "cuFileBatchIOGetStatus");
                    for (unsigned i = 0; i < returned; ++i) {
                        short_read |= events[i].ret != 4096;
                    }
                    completed += returned;
                    if (std::chrono::steady_clock::now() > deadline) {
                        throw std::runtime_error("GDS batch exceeded 60-second deadline");
                    }
                }
            } catch (...) {
                cuFileBatchIOCancel(batch_);
                throw;
            }
            if (short_read) throw std::runtime_error("GDS page read failed or returned fewer than 4096 bytes");
        }
        return pages.size() * 4096;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<PageReader>(module, "PageReader")
        .def(py::init<const std::string&, uintptr_t, size_t, int, py::object>())
        .def("read", &PageReader::read);
}
