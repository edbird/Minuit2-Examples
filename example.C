


#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"



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
        if(debug_level >= 2)
        {
            std::cout << __func__ << "(param[0]=" << param[0] << ")" << std::endl;
        }

        double x = param.at(0); // you can use range checking:
                                // NOT significantly slower

        // this is a parabola
        // how should param[0] be minimized? it should take the value 0.0
        // such that fval = 0.0
        // ... a very basic test
        double fval = 0.0;

        fval += (x * x);

        if(debug_level >= 3)
        {
            std::cout << "x * x = " << x * x << std::endl;
        }

        // add a gaussian constraint, if enabled
        if(constraint_enabled)
        {
            double c = std::pow((x - constraint_value) / constraint_error, 2.0);
            fval += c;

            if(debug_level >= 3)
            {
                std::cout << "c = " << c << std::endl;
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

    // we will have the ability to add a constraint
    public: // too lazy to implement interface
    mutable bool constraint_enabled = false;
    private:
    mutable double constraint_value = 5.0;
    mutable double constraint_error = 1.0;

};


void example_noconstraint()
{

    std::cout << "Example without constraint enabled" << std::endl;

    // initialization
    ROOT::Minuit2::MnUserParameterState theParameterStateBefore;
    ROOT::Minuit2::VariableMetricMinimizer theMinimizer;
    MinimizeFCN theFCN;

    // we might like to change the debug_level - permitted because "mutable"
    theFCN.debug_level = 2;

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
    
    const double initial_value = 3.0;
    const double initial_uncert = 0.5;
    theParameterStateBefore.Add(std::string(param_name_str), initial_value, initial_uncert);
    //theParameterState.Fix(std::string(param_name_str)); // you might need
    //theParameterState.SetLowerLimit(std::string(param_name_str)); // you might need
    //theParameterState.SetUpperLimit(std::string(param_name_str)); // you might need

    // save parameters for future reference
    std::vector<double> params_before = theParameterStateBefore.Params();
    std::vector<double> param_errs_before = theParameterStateBefore.Errors();

    double fval_before = theFCN.operator()(params_before);

    // exec
    ROOT::Minuit2::MnStrategy theStrategy(1); // no clue what it does
    ROOT::Minuit2::FunctionMinimum FCN_min = theMinimizer.Minimize(theFCN, theParameterStateBefore, theStrategy);

    // get results
    ROOT::Minuit2::MnUserParameterState theParameterStateAfter = FCN_min.UserParameters();
    // save parameters for future reference
    std::vector<double> params_after = theParameterStateAfter.Params();
    std::vector<double> param_errs_after = theParameterStateAfter.Errors();

    double fval_after = theFCN.operator()(params_after);

    // print
    std::cout << "fval changed from " << fval_before << " to " << fval_after << std::endl;
    std::cout << "param[0] changed from " << params_before.at(0) << " +- " << param_errs_before.at(0)
              << " to " << params_after.at(0) << " +- " << param_errs_after.at(0) << std::endl;
    
    std::cout << "done - it was easy" << std::endl;

    return 0;

}


void example_withconstraint()
{

    // NOTE:
    // How I set this up we minimize
    //
    // x * x + std::pow((x - cv) / err, 2.0) =
    // x * x + x * x / (err * err) - 2 * x * cv / err + cv * cv / (err * err)
    // for cv = 5.0, err = 1.0, we have:
    // 2 * x * x - 10 * x + 25
    // which has a minimum of 12.5 at x = 2.5 (check with Desmos)

    std::cout << "Example with constraint enabled" << std::endl;

    // initialization
    ROOT::Minuit2::MnUserParameterState theParameterStateBefore;
    ROOT::Minuit2::VariableMetricMinimizer theMinimizer;
    MinimizeFCN theFCN;

    // we might like to change the debug_level - permitted because "mutable"
    theFCN.debug_level = 3;

    // turn the constraint on
    theFCN.constraint_enabled = true;

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
    
    const double initial_value = 3.0;
    const double initial_uncert = 0.5;
    theParameterStateBefore.Add(std::string(param_name_str), initial_value, initial_uncert);
    //theParameterState.Fix(std::string(param_name_str)); // you might need
    //theParameterState.SetLowerLimit(std::string(param_name_str)); // you might need
    //theParameterState.SetUpperLimit(std::string(param_name_str)); // you might need

    // save parameters for future reference
    std::vector<double> params_before = theParameterStateBefore.Params();
    std::vector<double> param_errs_before = theParameterStateBefore.Errors();

    double fval_before = theFCN.operator()(params_before);

    // exec
    ROOT::Minuit2::MnStrategy theStrategy(1); // no clue what it does
    ROOT::Minuit2::FunctionMinimum FCN_min = theMinimizer.Minimize(theFCN, theParameterStateBefore, theStrategy);

    // get results
    ROOT::Minuit2::MnUserParameterState theParameterStateAfter = FCN_min.UserParameters();
    // save parameters for future reference
    std::vector<double> params_after = theParameterStateAfter.Params();
    std::vector<double> param_errs_after = theParameterStateAfter.Errors();

    double fval_after = theFCN.operator()(params_after);

    // print
    std::cout << "fval changed from " << fval_before << " to " << fval_after << std::endl;
    std::cout << "param[0] changed from (" << params_before.at(0) << " +- " << param_errs_before.at(0)
              << ") to (" << params_after.at(0) << " +- " << param_errs_after.at(0) << ")" << std::endl;
    
    std::cout << "done - it was easy" << std::endl;
    std::cout << std::endl;

    return 0;

}


void example()
{
    example_noconstraint();
    example_withconstraint();
}
