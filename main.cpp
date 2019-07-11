#include <iostream>

#include <fstream>
#include <iostream>
#include <chrono>
#include <XgboostPredictor.h>

int main() {

    std::cout.precision(9);
    using namespace std::chrono;
    long long int ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
    ).count();

    // testing xgboost model.
    std::string model_path = "../xgboost_model.txt";
    XgboostPredictor xgboostPredictor = XgboostPredictor(model_path, 2);

    long long int ms2 = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
    ).count();
    std::vector<double> input(27);
    input = {0.166704003,0.793647502,0.585092658,-0.836314314,-0.495913423,0.233769642,0.202316318,0.528412458,0.824529188,-0.85625963,-0.222778842,0.466035443,0.221061031,-0.715468667,-0.662749279,0.922008461,0.260256615,0.28664768,3.344844582,0.94415395,1.039894947,2.136208297,-1.528755739,-2.820955601,-0.750271,2.003565,-1.070326};
    std::vector<double> res = xgboostPredictor.Predict(input);

    long long int ms3 = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
    ).count();

    std::cout << "xgboost predict probability vector is: ["
              << res[0] << ", " << res[1]
              << "], load model using " << ms2 - ms
              << "(ms) predict using " << ms3 - ms2 << "(ms)"
              << std::endl;
}