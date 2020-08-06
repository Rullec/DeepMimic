#include "DrawObj.h"

void cDrawObj::Draw(const cSimObj *obj, cDrawUtil::eDrawMode draw_mode)
{
    cShape::eShape shape = obj->GetShape();
    switch (shape)
        {
        case cShape::eShapeBox:
            DrawBox(obj, draw_mode);
            break;
        case cShape::eShapePlane:
            DrawPlane(obj, draw_mode);
            break;
        case cShape::eShapeCapsule:
            DrawCapsule(obj, draw_mode);
            break;
        case cShape::eShapeSphere:
            DrawSphere(obj, draw_mode);
            break;
        case cShape::eShapeCylinder:
            DrawCylinder(obj, draw_mode);
            break;
        default:
            assert(false); // unsupported shape
            break;
        }
}

void cDrawObj::DrawBox(const cSimObj *box, cDrawUtil::eDrawMode draw_mode)
{
    DrawBox(box, tVector::Zero(), tVector::Ones(), draw_mode);
}

void cDrawObj::DrawBox(const cSimObj *box, const tVector &tex_coord_min,
                       const tVector &tex_coord_max,
                       cDrawUtil::eDrawMode draw_mode)
{
    assert(box->GetShape() == cShape::eShapeBox);
    tVector pos = box->GetPos();
    tVector size = box->GetSize();
    tVector axis;
    double theta;
    box->GetRotation(axis, theta);

    cDrawUtil::PushMatrixView();

    cDrawUtil::Translate(pos);
    cDrawUtil::Rotate(theta, axis);
    cDrawUtil::DrawBox(tVector::Zero(), size, tex_coord_min, tex_coord_max,
                       draw_mode);

    cDrawUtil::PopMatrixView();
}

void cDrawObj::DrawPlane(const cSimObj *plane, double size,
                         cDrawUtil::eDrawMode draw_mode)
{
    assert(plane->GetShape() == cShape::eShapePlane);
    tVector coeffs = plane->GetSize();
    cDrawUtil::DrawPlane(coeffs, size, draw_mode);
}

void cDrawObj::DrawCapsule(const cSimObj *cap, cDrawUtil::eDrawMode draw_mode)
{
    assert(cap->GetShape() == cShape::eShapeCapsule);
    tVector pos = cap->GetPos(); // part 中有get pos的函数，用来拿位置
    tVector size = cap->GetSize(); // 拿到"size",有什么用?
    double r = 0.5 * size[0];      // size的第一个参数是直径
    double h = size[1] - 2 * r;    // size的第一个参数是总长，那ok
    tVector axis;
    double theta;
    cap->GetRotation(
        axis, theta); // 物体的轴角表示(和欧拉角对应，是一种表示旋转的方法)

    cDrawUtil::PushMatrixView();

    cDrawUtil::Translate(pos);               // 平移变换
    cDrawUtil::Rotate(theta, axis);          // 轴角表示:旋转变换
    cDrawUtil::DrawCapsule(r, h, draw_mode); //　绘制胶囊

    cDrawUtil::PopMatrixView();
}

void cDrawObj::DrawSphere(const cSimObj *ball, cDrawUtil::eDrawMode draw_mode)
{
    assert(ball->GetShape() == cShape::eShapeSphere);
    tVector pos = ball->GetPos();
    tVector size = ball->GetSize();
    tVector axis;
    double theta;
    ball->GetRotation(axis, theta);
    double r = 0.5 * size[0];

    cDrawUtil::PushMatrixView();
    cDrawUtil::Translate(pos);
    cDrawUtil::Rotate(theta, axis);
    cDrawUtil::DrawSphere(r, draw_mode);
    cDrawUtil::PopMatrixView();
}

void cDrawObj::DrawCylinder(const cSimObj *cap, cDrawUtil::eDrawMode draw_mode)
{
    assert(cap->GetShape() == cShape::eShapeCylinder);
    tVector pos = cap->GetPos();
    tVector size = cap->GetSize();
    double r = 0.5 * size[0];
    double h = size[1];
    tVector axis;
    double theta;
    cap->GetRotation(axis, theta);

    cDrawUtil::PushMatrixView();

    cDrawUtil::Translate(pos);
    cDrawUtil::Rotate(theta, axis);
    cDrawUtil::DrawCylinder(r, h, draw_mode);

    cDrawUtil::PopMatrixView();
}