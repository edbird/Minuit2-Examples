
#include <vector>
#include <iostream>
#include <cmath>


#include "TH1.h"
#include "TRandom3.h"

#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"


// globals
// probably want to access elsewhere


class MinimizeFCN : public ROOT::Minuit2::FCNBase
{

    // NOTES:
    // All calculations MUST be done with "double" as far as possible
    // (conversion of input data from float to double should be done
    // as early as possible)
    // Reason: Minuit2 algorithm expects to be able to change parameters
    // by machine epsilon and observe difference in output fval obtained


    public:

    MinimizeFCN()
        : error_def{1.0} // change to 0.5 if maximizing LL
        , is_init{false}
    {
    }

    virtual
    ~MinimizeFCN()
    {
    }

    virtual
    double Up() const
    {
        return error_def;
    }
    

    ///////////////////////////////////////////////////////////////////////////
    // a function to init out minimization data
    // here we move the contents of TH1D bins into std::vector<double>
    // why? because std::vector<double> is significantly faster
    // my (more complex) analysis does not run fast enough without this
    // for a simple analysis it is not necessary
    // it is also not necessary as a first step and can be added later
    ///////////////////////////////////////////////////////////////////////////

    void init(TH1D *data,
              TH1D *mc_1,
              TH1D *mc_2,
              TH1D *mc_3)
    {
        if(is_init == true)
        {
            std::cout << "Fatal error in " << __func__ << "is_init=true" << std::endl;
            throw "is_init=true";
        }
        
        Int_t nbx_0 = data->GetNbinsX();
        Int_t nbx_1 = mc_1->GetNbinsX();
        Int_t nbx_2 = mc_2->GetNbinsX();
        Int_t nbx_3 = mc_3->GetNbinsX();

        if(nbx_0 ^ nbx_1)
        {
            std::cout << "Fatal error in " << __func__ << " number of bins in data and mc_1 not equal" << std::endl;
        }
        if(nbx_0 ^ nbx_2)
        {
            std::cout << "Fatal error in " << __func__ << " number of bins in data and mc_2 not equal" << std::endl;
        }
        if(nbx_0 ^ nbx_3)
        {
            std::cout << "Fatal error in " << __func__ << " number of bins in data and mc_3 not equal" << std::endl;
        }

        for(Int_t i = 1; i <= data->GetNbinsX(); ++ i)
        {
            double content_data = data->GetBinContent(i);
            data_data.push_back(content_data);

            double content_mc_1 = mc_1->GetBinContent(i);
            mc_1_data.push_back(content_mc_1);
            
            double content_mc_2 = mc_2->GetBinContent(i);
            mc_2_data.push_back(content_mc_2);
            
            double content_mc_3 = mc_3->GetBinContent(i);
            mc_3_data.push_back(content_mc_3);
        }

        is_init = true;

        if(debug_level >= 2)
        {
            std::cout << "done " << __func__ << std::endl;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    // operator()
    //
    // This function calculates "fval" (function value)
    // for function to be minimized
    //
    // In this example, we minimize a parabola: x^2
    //
    ///////////////////////////////////////////////////////////////////////////

    virtual
    double operator()(const std::vector<double> &param) const
    {
        if(is_init == false)
        {
            std::cout << "Fatal error in " << __func__ << " is_init=false" << std::endl;
            throw "is_init=false"; // BOOM
        }

        double A = param.at(0);
        double B = param.at(1);
        double C = param.at(2); // you can use range checking:
                                // NOT significantly slower
        if(debug_level >= 2)
        {
            std::cout << __func__ << "(A=" << A
                                  << ", B=" << B
                                  << ", C=" << C << ")" << std::endl;
        }

        // calc fval
        double fval = 0.0;

        // build total mc
        total_mc.clear();
        for(std::size_t i = 0; i < data_data.size(); ++ i)
        {
            double content = 0.0;
            content += A * mc_1_data[i];
            content += B * mc_2_data[i];
            content += C * mc_3_data[i];
            total_mc.push_back(content);
        }

        // loop over bins
        for(std::size_t i = 0; i < data_data.size(); ++ i)
        {
            double content_data = data_data[i];
            double content_mc = total_mc[i];
            double error_mc = std::sqrt(content_mc);
            double next = std::pow((content_data - content_mc) / error_mc, 2.0);
            fval += next;

            if(debug_level >= 3)
            {
                std::cout << "i=" << i << " next=" << next << std::endl;
            }
        }

        if(debug_level >= 1)
        {
            std::cout << __func__ << " operator()=" << fval << std::endl;
        }

        return fval;
    }

    protected:

    // you can define helper functions and encapsulate your minimization
    // related data in this class - however it is often easier to use
    // global variables if you require access to these variables elsewhere
    // in your code

    double error_def;

    // operator() is declared "const"
    // This means all members of this class must be const also
    // A "workaround" is to use mutable
    // mutable allows the variable to be changed despite the fact that a 
    // class method is declared "const"
    // You may not need it
    // To be technically correct, you should only declare variables as
    // mutable here if they are "invisible" from the point of view of the
    // user
    // For example: If rebuilding your data takes a long time when param[3]
    // is changed, then you may include something like
    //
    // // This variable tracks the last value of param[3]
    // mutable double param_3_last = -std::numeric_limits<double>::infinity();
    //
    // // If param[3] is different from last time, rebuild data
    // if(param[3] != param_3_last)
    // {
    //    rebuild_my_data();
    // }
    //
    // In this case, the user should not have direct access to "param_3_last"
    //
    // Another example:
    //
    // // Debug level flag, controls amount of detail produced by debug
    // // statements
    // int debug_level = 5;
    //
    // Maybe the user does have the ability to change "debug_level" via
    // an interface (method) - however it does not affect the "logical
    // state" of this class. (It does change the "binary state".)
    // Google for more details, this is largely a philosophical debate
    
    public: // too lazy to implement interface
    mutable int debug_level = 1;
    protected:

    mutable bool is_init;

    mutable std::vector<double> data_data;
    mutable std::vector<double> mc_1_data;
    mutable std::vector<double> mc_2_data;
    mutable std::vector<double> mc_3_data;
    mutable std::vector<double> total_mc;

};


void example()
{

    std::cout << "Example with pseudodata" << std::endl;

    // create some histograms
    TH1D *mc_1 = new TH1D("mc_1", "mc_1", 50, 0.0, 5.0);
    mc_1->Sumw2();
    mc_1->SetLineColor(kRed);
    mc_1->SetFillColor(kRed);

    TH1D *mc_2 = new TH1D("mc_2", "mc_2", 50, 0.0, 5.0);
    mc_2->Sumw2();
    mc_2->SetLineColor(kGreen);
    mc_2->SetFillColor(kGreen);

    TH1D *mc_3 = new TH1D("mc_3", "mc_3", 50, 0.0, 5.0);
    mc_3->Sumw2();
    mc_3->SetLineColor(kAzure);
    mc_3->SetFillColor(kAzure);

    TH1D *data = new TH1D("data", "data", 50, 0.0, 5.0);
    data->Sumw2();
    data->SetLineColor(kBlack);
    data->SetLineWidth(2.0);
    data->SetMarkerStyle(20);
    data->SetMarkerSize(1.0);


    // generate some pseudodata
    TRandom3 rng;
    for(int i = 0; i < 1000; ++ i)
    {
        double rand_gaus_1 = rng.Gaus(2.5, 1.0);
        double rand_gaus_2 = rng.Gaus(1.5, 0.25);
        double rand_exp_1 = rng.Exp(1.0);

        mc_1->Fill(rand_gaus_1);
        mc_2->Fill(rand_gaus_2);
        mc_3->Fill(rand_exp_1);
    }

    const double A_coeff = 3.0;
    const double B_coeff = 2.0;
    const double C_coeff = 1.0;

    for(Int_t i = 1; i <= mc_1->GetNbinsX(); ++ i)
    {
        double c1 = mc_1->GetBinContent(i);
        double c2 = mc_2->GetBinContent(i);
        double c3 = mc_3->GetBinContent(i);
        double c = A_coeff * c1 + B_coeff * c2 + C_coeff * c3;
        data->SetBinContent(i, c);
        data->SetBinError(i, std::sqrt(c));
    }

    // initialization
    ROOT::Minuit2::MnUserParameterState theParameterStateBefore;
    ROOT::Minuit2::VariableMetricMinimizer theMinimizer;
    MinimizeFCN theFCN;

    // we might like to change the debug_level - permitted because "mutable"
    theFCN.debug_level = 2;

    // init theFCN
    theFCN.init(data, mc_1, mc_2, mc_3);

    // initialization of parameters (I would put this bit in another function)
    int param_number = 0;
    TString param_number_str;
    param_number_str.Form("%i", param_number);
    TString param_name_str;
    param_name_str.Form("_%s_", param_number_str.Data()); // just an example
    // this is what I use, but you can change it
    // Personally I think Minuits idea of "naming" parameters is unhelpful
    // Sometimes I add "HARD", "SOFT" or "FREE" after the name to remind
    // myself which params are hard (fixed), free (free) or soft (gaussian
    // constrained)

    const double initial_value_A = 1.0;
    const double initial_uncert_A = 0.1;
    theParameterStateBefore.Add(std::string(param_name_str), initial_value_A, initial_uncert_A);
    theParameterStateBefore.SetLowerLimit(std::string(param_name_str), 0.0);
    //theParameterState.Fix(std::string(param_name_str)); // you might need
    //theParameterState.SetUpperLimit(std::string(param_name_str)); // you might need

    param_number = 1;
    param_number_str.Form("%i", param_number);
    param_name_str.Form("_%s_", param_number_str.Data());
    const double initial_value_B = 1.0;
    const double initial_uncert_B = 0.1;
    theParameterStateBefore.Add(std::string(param_name_str), initial_value_B, initial_uncert_B);
    theParameterStateBefore.SetLowerLimit(std::string(param_name_str), 0.0);
    param_number = 2;
    param_number_str.Form("%i", param_number);
    param_name_str.Form("_%s_", param_number_str.Data());
    const double initial_value_C = 1.0;
    const double initial_uncert_C = 0.1;
    theParameterStateBefore.Add(std::string(param_name_str), initial_value_C, initial_uncert_C);
    theParameterStateBefore.SetLowerLimit(std::string(param_name_str), 0.0);
    

    // save parameters for future reference
    std::vector<double> params_before = theParameterStateBefore.Params();
    std::vector<double> param_errs_before = theParameterStateBefore.Errors();

    double fval_before = theFCN.operator()(params_before);

    // draw
    mc_1->Scale(params_before.at(0));
    mc_2->Scale(params_before.at(1));
    mc_3->Scale(params_before.at(2));
    TCanvas *c_all_before = new TCanvas("c_all_before", "c_all_before");
    data->SetStats(0);
    data->SetTitle("");
    data->Draw("PE");
    THStack *mc_stack = new THStack("mc_stack", "mc_stack");
    mc_stack->Add(mc_3);
    mc_stack->Add(mc_1);
    mc_stack->Add(mc_2);
    mc_stack->Draw("histsame");
    data->Draw("Esame");
    //mc_1->Draw("histsame");
    //mc_2->Draw("histsame");
    //mc_3->Draw("histsame");
    c_all_before->Show();
    c_all_before->SaveAs("c_all_before.png");

    // exec
    ROOT::Minuit2::MnStrategy theStrategy(1); // no clue what it does
    ROOT::Minuit2::FunctionMinimum FCN_min = theMinimizer.Minimize(theFCN, theParameterStateBefore, theStrategy);

    // get results
    ROOT::Minuit2::MnUserParameterState theParameterStateAfter = FCN_min.UserParameters();
    // save parameters for future reference
    std::vector<double> params_after = theParameterStateAfter.Params();
    std::vector<double> param_errs_after = theParameterStateAfter.Errors();

    double fval_after = theFCN.operator()(params_after);

    // draw
    mc_1->Scale(params_after.at(0) / params_before.at(0));
    mc_2->Scale(params_after.at(1) / params_before.at(1));
    mc_3->Scale(params_after.at(2) / params_before.at(2));
    TCanvas *c_all_after = new TCanvas("c_all_after", "c_all_after");
    data->SetStats(0);
    data->SetTitle("");
    data->Draw("PE");
    THStack *mc_stack_2 = new THStack("mc_stack_2", "mc_stack_2");
    mc_stack_2->Add(mc_3);
    mc_stack_2->Add(mc_1);
    mc_stack_2->Add(mc_2);
    mc_stack_2->Draw("histsame");
    data->Draw("Esame");
    //mc_1->Draw("histsame");
    //mc_2->Draw("histsame");
    //mc_3->Draw("histsame");
    c_all_after->Show();
    c_all_after->SaveAs("c_all_after.png");

    // print
    std::cout << "fval changed from " << fval_before << " to " << fval_after << std::endl;
    std::cout << "param[0] changed from " << params_before.at(0) << " +- " << param_errs_before.at(0)
              << " to " << params_after.at(0) << " +- " << param_errs_after.at(0) << std::endl;
    std::cout << "param[1] changed from " << params_before.at(1) << " +- " << param_errs_before.at(1)
              << " to " << params_after.at(1) << " +- " << param_errs_after.at(1) << std::endl;
    std::cout << "param[2] changed from " << params_before.at(2) << " +- " << param_errs_before.at(2)
              << " to " << params_after.at(2) << " +- " << param_errs_after.at(2) << std::endl;

    params_after[0] = params_after[0] + param_errs_after[0];
    double fval_after_2 = theFCN.operator()(params_after);
    std::cout << "fval_after_2=" << fval_after_2 << " fval_after=" << fval_after << std::endl;
    
    std::cout << "done - it was easy" << std::endl;

    return 0;

}



