# xgboost-predictor-cpp

## Getting started

- The only thing to do is include the `XgboostPredictor.h` file to your project and add the `XgboostPredictor.cpp` file to your source file list.

- How to invoke
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