//
// Created by ljf on 2020/7/3.
//

#include "ShapeMotionMemPool.h"
#include <cassert>

ShapeMotionMemPool::ShapeMotionMemPool(unsigned int length):length(length) {
    assert(length != 0);
}
