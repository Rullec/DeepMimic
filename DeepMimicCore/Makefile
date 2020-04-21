BIN=Main
OBJ_DIR = objs
MKDIR = mkdir -p
OS=$(shell uname)
$(info [log] Current system is [${OS}])

# EIGEN_DIR = ../../libraries/eigen
# BULLET_INC_DIR = ../../libraries/bullet3/src
EIGEN_DIR = ../../include/
BULLET_INC_DIR = ../../bullet3/src

ifeq ($(OS),Darwin)
	CONDA_DIR=../../../anaconda3
else
	CONDA_DIR=../../../.conda
endif

# PYTHON_INC = /usr/include/python3.6m
# PYTHON_LIB = /usr/lib/ -lpython3.6m
PYTHON_INC = $(CONDA_DIR)/envs/spinningup/include/python3.6m
PYTHON_LIB = $(CONDA_DIR)/envs/spinningup/lib -lpython3.6m
export LD_LIBRARY_PATH=$(CONDA_DIR)/envs/spinningup/lib:$(CONDA_DIR)/lib:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$(LIBRARY_PATH):/usr/local/lib

INC = -I./ \
	-I$(EIGEN_DIR) \
	-I$(BULLET_INC_DIR)


# normal
# LIBS = -lGLEW -lGL -lGLU -lglut -lBulletDynamics -lBulletCollision -lLinearMath -lBullet3Common -lBulletInverseDynamics -lBulletInverseDynamicsUtils -lm -lstdc++

# after
ifeq ($(OS), Darwin)
	BULLET_CMAKE_DIR=/Users/xudong/Projects/bullet3/build_cmake
	LIB_SUFFIX=.dylib
else
	BULLET_CMAKE_DIR=/home/xudong/Projects/bullet3/build_cmake
	LIB_SUFFIX=.so
endif

define BULLET_LIBS
	$(BULLET_CMAKE_DIR)/src/BulletDynamics/libBulletDynamics$(LIB_SUFFIX)\
	$(BULLET_CMAKE_DIR)/src/BulletCollision/libBulletCollision$(LIB_SUFFIX)\
	$(BULLET_CMAKE_DIR)/src/LinearMath/libLinearMath$(LIB_SUFFIX)\
	$(BULLET_CMAKE_DIR)/src/Bullet3Common/libBullet3Common$(LIB_SUFFIX)\
	$(BULLET_CMAKE_DIR)/src/BulletInverseDynamics/libBulletInverseDynamics$(LIB_SUFFIX)\
	$(BULLET_CMAKE_DIR)/Extras/InverseDynamics/libBulletInverseDynamicsUtils$(LIB_SUFFIX)
endef

ifeq ($(OS), Darwin)
	LIBS=-lGLEW -framework OpenGL -framework GLUT $(BULLET_LIBS) -lm -lstdc++
else
	LIBS = -lGLEW -lGL -lGLU -lglut $(BULLET_LIBS) -lm -lstdc++
endif
MACROS = --define-macro BT_USE_DOUBLE_PRECISION

CC = clang++
# CC = g++
CFLAGS = -std=c++11 -O3 -Wall -fPIC -Wdeprecated-declarations

SRC_DIRS = util/ \
	util/json/ \
	anim/ \
	sim/ \
	render/ \
	render/lodepng/ \
	scenes/ \
	scenes/arm/ \
	scenes/particle/ \
	scenes/pendulum/

SRC_FILES = $(BIN).cpp DeepMimicCore.cpp $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)*.cpp))
OUT_DIRS = $(addprefix $(OBJ_DIR)/, $(SRC_DIRS))
OBJ = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRC_FILES)))


default: build

build: $(BIN)

print:
	$(OUT_DIRS)

python: $(OUT_DIRS) $(OBJ)
	swig -c++ -python DeepMimicCore.i
	$(CC) -c -g $(CFLAGS) $(MACROS) $(INC) $(LIBS) DeepMimicCore.cpp DeepMimicCore_wrap.cxx -I$(PYTHON_INC)
	$(CC) -dynamiclib -flat_namespace -undefined suppress $(CFLAGS) $(MACROS) $(OBJ) $(LIBS) DeepMimicCore_wrap.o -o _DeepMimicCore.so -L$(PYTHON_LIB)

$(OBJ_DIR)/%.o: %.cpp
	$(CC) -c -g $(CFLAGS) $(MACROS) $(INC) -o $@ $<

$(BIN): $(OUT_DIRS) $(OBJ)
	$(CC) -g $(CFLAGS) $(MACROS) $(OBJ) $(LIBS) -o $@

$(OUT_DIRS):
	$(MKDIR) $(OUT_DIRS)
	
clean:
	- rm -f -r $(OUT_DIRS) $(BIN) $(OBJ) $(OBJ_DIR) DeepMimicCore_wrap.cxx _DeepMimicCore$(LIB_SUFFIX)