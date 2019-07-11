# xgboost-predictor-cpp

## Getting started

- The only thing to do:
```
Including the `XgboostPredictor.h` file to your project and add the `XgboostPredictor.cpp` file to your source file list.
```

- How to invoke:
```
#include <XgboostPredictor.h>

// load model
std::string model_path = "here is your xgboost model path";
// here is the total number of your predict classes.
int class_num = 2;
XgboostPredictor xgboostPredictor = XgboostPredictor(model_path, class_num);

// predict
std::vector<double> input = {here is you features};
// prediction result is the probability of each category.
std::vector<double> res = xgboostPredictor.Predict(input);
```

- more test:
```
#include <iostream>
#include <fstream>
#include <chrono>
#include <XgboostPredictor.h>

std::cout.precision(9);
using namespace std::chrono;
long long int ms = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();

// loading xgboost model.
std::string model_path = "../xgboost_model.txt";
XgboostPredictor xgboostPredictor = XgboostPredictor(model_path, 2);
long long int ms2 = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();

// predict
std::vector<double> input(27);
input = {0.166704003,0.793647502,0.585092658,-0.836314314,-0.495913423,0.233769642,0.202316318,0.528412458,0.824529188,-0.85625963,-0.222778842,0.466035443,0.221061031,-0.715468667,-0.662749279,0.922008461,0.260256615,0.28664768,3.344844582,0.94415395,1.039894947,2.136208297,-1.528755739,-2.820955601,-0.750271,2.003565,-1.070326};
std::vector<double> res = xgboostPredictor.Predict(input);
long long int ms3 = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();

std::cout << "xgboost predict probability vector is: ["
          << res[0] << ", " << res[1]
          << "], load model using " << ms2 - ms
          << "(ms) predict using " << ms3 - ms2 << "(ms)"
          << std::endl;
```

`result is:`

```
xgboost predict probability vector is: [0.692828893, 0.307171107], load model using 1358(ms) predict using 1(ms)
```


## TODO List:

**Now this project only support Xgboost models trained with 'multi:softprob' objective, and dump with txt type.**

- [X] support txt type.
- [ ] support json type.
- [ ] support binary type.
- [ ] support different objective function.