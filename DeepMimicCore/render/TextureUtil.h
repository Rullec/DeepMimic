#pragma once

#include <GL/glew.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <fstream>

#include "util/MathUtil.h"

class cTextureUtil
{
public:
    static void CreateFrameBuffer(GLuint &buffer_obj, GLuint &texture,
                                  GLuint &depth_stencil, int width, int height,
                                  int depth, int channels, GLenum format,
                                  GLenum type, bool mipmaps = false);
    static void DeleteFrameBuffer(GLuint &buffer_obj, GLuint &texture,
                                  GLuint &depth_stencil);
    static tVector ReadTexel(int x, int y, int w, int h,
                             const std::vector<GLfloat> &data);
    static int GetNumChannels(GLint channels);
};
