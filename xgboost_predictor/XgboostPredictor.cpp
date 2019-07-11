//
// Created by yangcheng on 2019/7/4.
//

#include "XgboostPredictor.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include "cassert"

XgboostPredictor::~XgboostPredictor() {};

/**
 * xgboost 模型读取, 解压, 解析成shared_ptr<XTree>.
 * @param model_path
 */
XgboostPredictor::XgboostPredictor(std::string &model_path, int class_num) {

    this->class_num_ = class_num;
    XTree_map_ptr xtree_map_ptr = std::make_shared<std::unordered_map<int, XTree_ptr>>();

    std::ifstream infile(model_path);
    assert(infile.is_open());
    std::string line;
    std::getline(infile, line);

    int boost_id = 0;
    while (std::getline(infile, line)) {
        if (line.find("booster") != std::string::npos) {
            boost_id += 1;
            this->XTrees.push_back(xtree_map_ptr);
            xtree_map_ptr.reset(new std::unordered_map<int, XTree_ptr>());
        } else {
            int node_id = std::stoi(this->split(line, ":")[0]);
            XTree xtree = this->detectTrees(line);
            XTree_ptr xtree_ptr = std::make_shared<XTree>(xtree);
            xtree_map_ptr->emplace(std::make_pair(node_id, xtree_ptr));
        }
    }
    this->XTrees.push_back(xtree_map_ptr);
}

/**
 * detect tree node and convert it into XTree structure.
 *
 * @param model_line: file read line.
 * @return XTree
 */
XTree XgboostPredictor::detectTrees(std::string &model_line) {

    if (model_line.find("leaf") == std::string::npos) {

        // e.g "0:[f5<0.999779344] yes=1,no=2,missing=1" => "0:[f5<0.999779344]" "yes=1,no=2,missing=1"
        std::vector<std::string> feature_node = this->split(model_line, " ");
        // e.g "0:[f5<0.999779344]" => "0:[f5" "0.999779344]"
        std::vector<std::string> feature_s = this->split(feature_node[0], "<");

        // feature idx; e.g "0:[f5" => "0:" "f5"
        std::string feature_idx_s = this->split(feature_s[0], "[")[1];
        feature_idx_s.erase(0, 1);
        int feature_idx = std::stoi(feature_idx_s);

        // split condition; e.g "0.999779344]" => 0.999779344
        std::string split_condition_s = feature_s[1];
        split_condition_s.pop_back();
        double split_condition = std::stod(split_condition_s);

        // yes/no/missing
        std::vector<std::string> node_s = this->split(feature_node[1], ",");
        int yes_node = std::stoi(this->split(node_s[0], "=")[1]);
        int no_node = std::stoi(this->split(node_s[1], "=")[1]);
        int missing_node = std::stoi(this->split(node_s[2], "=")[1]);
        return {feature_idx, split_condition, yes_node, no_node, missing_node};

    } else {

        double leaf_value = std::stof(this->split(model_line, "=")[1]);
        return {leaf_value};

    }

}

/**
 * predict the vector belong to which class.
 * Note: typedef std::shared_ptr<XTree> XTree_ptr;
 *       typedef std::shared_ptr<std::unordered_map<int, XTree_ptr>> XTree_map_ptr;
 *       std::vector<XTree_map_ptr> XTrees;
 *
 * @param current_input
 * @return
 */
std::vector<double> XgboostPredictor::predictTrees(std::vector<double> &current_input) const {

    std::vector<double> res(this->class_num_);
    int tag = 0;
    for (auto &tree_map : this->XTrees) {

        int class_idx = tag % this->class_num_;
        tag += 1;

        int node_id = 0;
        while (tree_map->find(node_id) != tree_map->end()) {

            XTree_ptr tree = (*tree_map)[node_id];

            if (tree->feature_idx_ == -1) {
                res[class_idx] += tree->leaf_weight_;
                break;
            } else {
                double feature = current_input[tree->feature_idx_];
                if (feature != this->missing_feature) {
                    if (feature < tree->split_condition_) {
                        node_id = tree->left_node_;
                    } else {
                        node_id = tree->right_node_;
                    }
                } else {
                    node_id = tree->miss_node_;
                }
            }
        }
    }
    return res;
}

/**
 * predict current data using the xgboost model.
 *
 * @param data, input vector.
 * @return preidct probability vector for each class according to input training class order.
 */
std::vector<double> XgboostPredictor::Predict(std::vector<double> &data) const {

    std::vector<double> res = this->predictTrees(data);
    std::vector<double> predict_prob;

    double sum_all = 0;
    for(double &v : res) sum_all += exp(v);
    for(double &v2: res) predict_prob.push_back(exp(v2) / sum_all);

    return predict_prob;
}


/**
 * string split.
 *
 * @param s: target split string.
 * @param split_tag: target split tag.
 * @return
 */
std::vector<std::string> XgboostPredictor::split(const std::string &s, std::string split_tag) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(split_tag);
    pos1 = 0;

    std::vector<std::string> v;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + split_tag.size();
        pos2 = s.find(split_tag, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
    return v;
}
