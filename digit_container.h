#ifndef DCONT
#define DCONT

#include "canvas.h"
#include <string>

class digit_container
{
private:
    BMP_Image digit_img;
    int h = 0, w;
    uint8_t** pixels;
public:
    digit_container(){}
    digit_container(int h, int w)
    {
        this->h = h;
        this->w = w;
        pixels = new uint8_t*[h];
        for (int i = 0; i < h; ++i)
        {
            pixels[i] = new uint8_t[w];
        }
        digit_img = BMP_Image(28, 28);
    }
    digit_container& operator=(const digit_container& equals_to)
    {
        for (int i = 0; i < h; ++i) delete[] pixels[i];
        delete[] pixels;
        h = equals_to.h;
        w = equals_to.w;
        pixels = new uint8_t * [h];
        for (int i = 0; i < h; ++i) pixels[i] = new uint8_t[w];
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                pixels[i][j] = equals_to.pixels[i][j];
            }
        }
        digit_img = equals_to.digit_img;
        return *this;
    }
    digit_container(const digit_container& copy)
    {
        h = copy.h;
        w = copy.w;
        digit_img = copy.digit_img;
        pixels = new uint8_t * [h];
        for (int i = 0; i < h; ++i) pixels[i] = new uint8_t[w];
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                pixels[i][j] = copy.pixels[i][j];
            }
        }
    }
    digit_container(std::string filename, int h, int w)
    {
        if (filename.find(".bmp") == std::string::npos)
        {
            std::cout << "Error: unsupported resolution." << std::endl;
            exit(-1);
        }
        this->h = h; this->w = w;
        digit_img.load(filename);
        if (w != digit_img.get_width() || h != digit_img.get_height())
        {
            std::cout << "Error: wrong shape." << std::endl;
            exit(-1);
        }
        pixels = new uint8_t * [h];
        for (int i = 0; i < h; ++i) pixels[i] = new uint8_t[w];
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                Pixel P = digit_img.get_pixel(x, y);
                if ((P.R + P.G + P.B) <= 380) // ~255*3 / 2 -- noise removement
                {
                    pixels[y][x] = 1;
                }
                else
                {
                    pixels[y][x] = 0;
                }
            }
        }
    }
    ~digit_container()
    {
        if (pixels != nullptr)
        {
            for (int i = 0; i < h; ++i)
            {
                delete[] pixels[i];
            }
            delete[] pixels;
            pixels = nullptr;
        }
    }
    bool get(int x, int y)
    {
        if (x >= w || y >= h || x < 0 || y < 0)
        {
            std::cout << "Error: wrong pixel ID." << std::endl;
            exit(-1);
        }
        return static_cast<bool>(pixels[y][x]);
    }
    void set(int x, int y, uint8_t val)
    {
        (val > 120) ? val = 255 : val = 0;
        pixels[y][x] = val;
        digit_img.put_pixel(x, y, 255-val, 255-val, 255-val);
    }
    void save(std::string filename)
    {
        digit_img.save(filename);
    }
    std::pair<int, int> dim() const
    {
        return std::pair<int, int>(h, w);
    }
};


#endif // DCONT
