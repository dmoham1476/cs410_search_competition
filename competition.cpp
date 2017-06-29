/**
 * @file competition.cpp
 * @author Hussein Hazimeh
 * Built based on query-runner.cpp which is written by Sean Massung
 */

/**deepam2 submission for the search competition

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "meta/corpus/document.h"
#include "meta/index/eval/ir_eval.h"
#include "meta/index/inverted_index.h"
#include "meta/index/ranker/ranker_factory.h"
#include "meta/index/score_data.h"
#include "meta/parser/analyzers/tree_analyzer.h"
#include "meta/sequence/analyzers/ngram_pos_analyzer.h"
#include "meta/util/time.h"
#include "meta/index/ranker/okapi_bm25.h"
#include "meta/index/ranker/jelinek_mercer.h"
#include "meta/index/ranker/pivoted_length.h"
#include "meta/index/ranker/dirichlet_prior.h"

using namespace meta;

// mptf2ln_ranker is a class that implements the mptf2ln ranking function. It is derived from the base class ranker.
class mptf2ln_ranker: public index::ranker
{
private: // s_, mu_, alpha_ and lambda_ are the parameters of mptf2ln
    float s_ = 0.2;
    float mu_ = 2000;
    float alpha_ = 0.3;
    float lambda_ = 0.7;

public:
//    const static std::string id;
    const static util::string_view id;
    mptf2ln_ranker(); // Default Constructor
    mptf2ln_ranker(float s, float mu, float alpha, float lambda);
    mptf2ln_ranker(std::istream& in);
 // Constructor which can set parameter values
    void set_param(float s, float mu, float alpha, float lambda){s_ = s; mu_ = mu; alpha_ = alpha; lambda_ = lambda;}; // Setter
    float score_one(const index::score_data&) override; // Calculates the score for a single matched term
void save(std::ostream& out) const override;
};

const util::string_view mptf2ln_ranker::id = "mptf2ln"; // Used to identify the
mptf2ln_ranker::mptf2ln_ranker(){}
mptf2ln_ranker::mptf2ln_ranker(float s, float mu, float alpha, float lambda) : s_{s}, mu_{mu}, alpha_{alpha}, lambda_{lambda} {}
mptf2ln_ranker::mptf2ln_ranker(std::istream& in)
    : s_{io::packed::read<float>(in)}, mu_{io::packed::read<float>(in)}, alpha_{io::packed::read<float>(in)}, lambda_{io::packed::read<float>(in)}{}

void mptf2ln_ranker::save(std::ostream& out) const {
  io::packed::write(out, id);
  io::packed::write(out, s_);
  io::packed::write(out, mu_);
  io::packed::write(out, alpha_);
  io::packed::write(out, lambda_);

}

float mptf2ln_ranker::score_one(const index::score_data& sd)
{
    /*
    This function is called for each matched term between the query and the document.
    The function's argument is a struct that contains important information about
    the matched term. For example, sd.doc_term_count gives the # of occurrences of
    the term in the document.
    */
    float doc_len = sd.idx.doc_size(sd.d_id); // Current document length
    float avg_dl = sd.avg_dl; // Average document length in the corpus
    float tf = sd.doc_term_count; // Raw term count in the document
    float df = sd.doc_count; // number of docs that term appears in
    float pc = static_cast<float>(sd.corpus_term_count) / sd.total_terms; //this is p(t/C)

    float s = s_; // mptf2ln's parameter
    float mu = mu_; // mptf2ln's parameter
    float alpha = alpha_; // mptf2ln's parameter
    float lambda = lambda_; // mptf2ln's parameter

    float tfok = 2.2*tf/(1.2+tf); // okapi tf term
    float idfpiv = std::log((sd.num_docs + 1.0)/df);
    float tfidfdir = std::log(1.0 + tf/(mu*pc));
    float lnpiv = 1 - s + s*doc_len/avg_dl;

    float tfidf2 = alpha*tfok*idfpiv + (1.0 - alpha)*tfidfdir;

    float score = sd.query_term_weight*tfidf2/std::pow(lnpiv,lambda);

    return score; // Change 0 to the final score you calculated
}



void mptf2ln_tune (const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval, float & alpha, float & lambda, float & s, float & mu, int & Q)
{

    float alphavalues [6] = {0.3, 0.35, 0.4, 0.45, 0.5, 0.6}; // Different values for the parameter alpha
    float lambdavalues [6] = {0.7, 0.75, 0.8, 0.85, 0.9, 0.95}; // Different values for the parameter lambda
    float svalues [5] = {0.2, 0.25, 0.3, 0.35, 0.4}; // Different values for the parameter alpha
    float muvalues [3] = {500,1000,2000.0}; // Different values for the parameter lambda
    unsigned int Qvalues [5] = {10, 20, 30, 50, 100};
    float maxmap = 0; // Stores the current maximum MAP value
    float smax = 0.2;
    float mumax = 500;
    float alphamax = 0.3; // Stores the current optimal alpha (i.e. c that achieves max MAP) - Ignore the initial value
    float lambdamax = 0.7; // Stores the current optimal lambda - Ignore the initial value
    int Qmax = 50;
  
    auto ranker = make_unique<mptf2ln_ranker>(); // creates a pointer to a mptf2ln_ranker instance

    for (int i=0 ; i<6 ; i++) // Loops over all alpha values
    {
        for (int j=0 ; j<6 ; j++) // Loops over all lambda values
         {
            for (int k=0 ; k<5 ; k++) //s values
                {
                    for (int l=0 ; l<3; l++) //mu values
                    {
                        for (int m=0 ; m<5 ; m++)
                            {
                                ranker->set_param(svalues[k], muvalues[l], alphavalues[i],lambdavalues[j]); // Sets the parameters of ranker to the current values of alpha and lambda


                                for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query) // Iterates over all queries in allqueries
                                {
                                    auto ranking = ranker->score(*idx, *query, Qvalues[m]); // Returns a ranked list of the top 1000 documents for the current query
                                    eval.avg_p(ranking,(*query).id(),Qvalues[m]); // eval.avg_p stores the value of average precision for the current query in the instance eval
                                }
                                if (eval.map() > maxmap) // Updates maxmap, alphamax, lambdamax if the current map, which is equal to eval.map(), is greater than maxmap
                                {
                                    // You should only change the values of the following three assignments
                                    maxmap = eval.map(); // Change 0 to the correct value DONE
                                    alphamax = alphavalues[i]; // Change 0 to the correct value DONE
                                    lambdamax = lambdavalues[j]; // Change 0 to the correct value DONE
                                    smax = svalues[k];
                                    mumax = muvalues[l];
                                    Qmax = Qvalues[m];
                                }


                                eval.reset_stats(); // Deletes all the average precision values stored in eval to allow correct calculation of MAP for the next parameter combination

                            }
                    }
                }
         }
    }
    std::cout<<"Max MAP = "<< maxmap << " achieved by " << "s = " << smax << ", mu = " << mumax << ", alpha = " << alphamax << ", lambda = " << lambdamax << std::endl;
    alpha = alphamax; // Returns the best c value to the calling function
    lambda = lambdamax; // Returns the best lambda value to the calling function
    s = smax;
    mu = mumax;
    Q = Qmax;
}



namespace meta{
namespace index{
template <>
std::unique_ptr<ranker>make_ranker<mptf2ln_ranker>(
        const cpptoml::table & config) // Used by mptf2ln_ranker to read the parameters c and lambda from config.toml - You can ignore it
{
    float s = 0.2;
    if (auto s_file = config.get_as<double>("s"))
        s = *s_file;

    float mu = 2000;
    if (auto mu_file = config.get_as<double>("mu"))
        mu = *mu_file;

    float alpha = 0.3;
    if (auto alpha_file = config.get_as<double>("alpha"))
        alpha = *alpha_file;

    float lambda = 0.7;
    if (auto lambda_file = config.get_as<double>("lambda"))
        lambda = *lambda_file;

    return make_unique<mptf2ln_ranker>(s, mu, alpha, lambda);
}

}
}

/*
class new_ranker: public index::ranker
{
private: // Change the parameters to suit your ranking function
    float param1_ = 0;
    float param2_ = 0;

public:
    const static util::string_view id;
    new_ranker(); // Default Constructor
    new_ranker(float param1, float param2); // Constructor
    new_ranker(std::istream& in);
    void set_param(float param1, float param2){param1_ = param1; param2_ =
param2;}; // Sets the parameters
    void save(std::ostream& out) const override;
    float score_one(const index::score_data&); // Calculates the score for one
matched term
};

const util::string_view new_ranker::id = "newranker"; // Used to identify the
ranker in config.toml
new_ranker::new_ranker(){}
new_ranker::new_ranker(float param1, float param2) : param1_{param1},
param2_{param2} {}

new_ranker::new_ranker(std::istream& in)
    : param1_{io::packed::read<float>(in)},
      param2_{io::packed::read<float>(in)}
{
    // nothing
}

void new_ranker::save(std::ostream& out) const
{
    io::packed::write(out, id);
    io::packed::write(out, param1_);
    io::packed::write(out, param2_);
}

float new_ranker::score_one(const index::score_data& sd)
{
    // Implement your scoring function here

   return 0;

}


namespace meta{
namespace index{
template <>
std::unique_ptr<ranker>make_ranker<new_ranker>(
        const cpptoml::table & config) // Used by new_ranker to read the
parameters param1 and param2 from config.toml
{
    float param1 = 0; // Change to the default parameter value
    if (auto param1_file = config.get_as<float>("param1"))
        param1 = *param1_file;

    float param2 = 0; // Change to the default parameter value
    if (auto param2_file = config.get_as<float>("param2"))
        param2 = *param2_file;

    return make_unique<new_ranker>(param1, param2);
}

}
}

*/

void bm25_tune (const std::shared_ptr<index::dblru_inverted_index> &idx,
		std::vector<corpus::document> &allqueries, 
		index::ir_eval &eval, float &k1, float &b, float &k3) {

    float k1values [12] = {1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0}; // Different values for the parameter k1
    float bvalues [12] = {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5, 2.0, 3.0}; // Different values for the parameter b
    float k3values [7] = {200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0};// Different values for parameter k3
    float maxmap = 0; // Stores the current maximum MAP value
    float k1max = 1.2;
    float bmax = 0.75;
    float k3max = 500.0;
    std::ofstream writeout;

    for (int i=0 ; i<12 ; i++) // Loops over all k1 values
    {
        for (int j=0 ; j<12 ; j++) // Loops over all b values
         {
            for (int k=0 ; k<7 ; k++) // Loops over all k3 values
            {
		index::okapi_bm25 ranker{k1values[i], bvalues[j], k3values[k]};	
                std::cout << "Tuning for parameters : " << "k1 = " << k1values[i] << "; b = " << bvalues[j] << "; k3 = " << k3values[k] << std::endl;
                for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query) // Iterates over all queries in allqueries
                {
                    auto ranking = ranker.score(*idx, *query, 50); // Returns a ranked list of the top 1000 documents for the current query
                    eval.avg_p(ranking,(*query).id(),50); // eval.avg_p stores the value of average precision for the current query in the instance eval
                }
                writeout <<  k1values[i] << "\t" << bvalues[j] << "\t" << k3values[k] << "\t" << eval.map() << "\n";


                if (eval.map() > maxmap) // Updates maxmap, cmax, lambdamax if the current map, which is equal to eval.map(), is greater than maxmap
                {
                    // You should only change the values of the following three assignments
                    maxmap = eval.map(); // Change 0 to the correct value DONE
                    k1max = k1values[i]; // Change 0 to the correct value DONE
                    bmax = bvalues[j]; // Change 0 to the correct value DONE
                    k3max = k3values[k];
                }

                eval.reset_stats(); // Deletes all the average precision values stored in eval to allow correct calculation of MAP for the next parameter combination
            }
         }
    }

    writeout.close();
    std::cout<<"Max MAP = "<< maxmap << " achieved by " << "k1 = " << k1max << ", b = " << bmax << ", k3 = " << k3max << std::endl; // Prints to the standard ouput
    k1 = k1max; // Returns the best c value to the calling function
    b = bmax;
    k3 = k3max; // Returns the best lambda value to the calling function
}

void jm_tune (const std::shared_ptr<index::dblru_inverted_index> &idx,
		std::vector<corpus::document> &allqueries, 
		index::ir_eval &eval, float &l) {

    float lambda[10] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1}; // Different values for the parameter k1
    float maxmap = 0; // Stores the current maximum MAP value
    float lambdamax = 0.001;
    std::ofstream writeout;

  
    for (int i=0 ; i<10 ; i++) // Loops over all k1 values
    {
	index::jelinek_mercer ranker{lambda[i]};	
        std::cout << "Tuning for parameters : " << "lambda = " << lambda[i] << std::endl;
        for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query) // Iterates over all queries in allqueries
                {
                    auto ranking = ranker.score(*idx, *query, 50); // Returns a ranked list of the top 1000 documents for the current query
                    eval.avg_p(ranking,(*query).id(),50); // eval.avg_p stores the value of average precision for the current query in the instance eval
                }
                writeout <<  lambda[i] << "\t" << eval.map() << "\n";

			std::cout << eval.map() << "\t" ;
                if (eval.map() > maxmap) // Updates maxmap, cmax, lambdamax if the current map, which is equal to eval.map(), is greater than maxmap
                {
                    // You should only change the values of the following three assignments
                    maxmap = eval.map(); // Change 0 to the correct value DONE
                    lambdamax = lambda[i];
                }

                eval.reset_stats(); // Deletes all the average precision values stored in eval to allow correct calculation of MAP for the next parameter combination
            }

    writeout.close();
    std::cout<<"Max MAP = "<< maxmap << " achieved by " << "lambda = " << lambdamax << std::endl; // Prints to the standard ouput
    l = lambdamax; // Returns the best lambda value to the calling function
}

void pivot_tune (const std::shared_ptr<index::dblru_inverted_index> &idx,
		std::vector<corpus::document> &allqueries, 
		index::ir_eval &eval, float &s_) {

    float s[11] = {0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3}; // Different values for the parameter k1
    float maxmap = 0; // Stores the current maximum MAP value
    float smax = 0.18;
    std::ofstream writeout;

    for (int i=0 ; i<11 ; i++) // Loops over all k1 values
    {
	index::pivoted_length ranker{s[i]};	
        std::cout << "Tuning for parameters : " << "s = " << s[i] << std::endl;
        for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query) // Iterates over all queries in allqueries
                {
                    auto ranking = ranker.score(*idx, *query, 50); // Returns a ranked list of the top 1000 documents for the current query
                    eval.avg_p(ranking,(*query).id(),50); // eval.avg_p stores the value of average precision for the current query in the instance eval
                }
                writeout <<  s[i] << "\t" << eval.map() << "\n";

			std::cout << eval.map() << "\n" ;
                if (eval.map() > maxmap) // Updates maxmap, cmax, smax if the current map, which is equal to eval.map(), is greater than maxmap
                {
                    // You should only change the values of the following three assignments
                    maxmap = eval.map(); // Change 0 to the correct value DONE
                    smax = s[i];
                }

                eval.reset_stats(); // Deletes all the average precision values stored in eval to allow correct calculation of MAP for the next parameter combination
            }

    writeout.close();
    std::cout<<"Max MAP = "<< maxmap << " achieved by " << "s = " << smax << std::endl; // Prints to the standard ouput
    s_ = smax; // Returns the best s value to the calling function
}

void dirichlet_tune (const std::shared_ptr<index::dblru_inverted_index> &idx,
		std::vector<corpus::document> &allqueries, 
		index::ir_eval &eval, float &mu_) {

    float mu[6] = {250,500,550,600,650,700}; // Different values for the parameter k1
    float maxmap = 0; // Stores the current maximum MAP value
    float mumax = 1000;
    std::ofstream writeout;

    for (int i=0 ; i<6 ; i++) // Loops over all k1 values
    {
	index::dirichlet_prior ranker{mu[i]};	
        std::cout << "Tuning for parameters : " << "mu = " << mu[i] << std::endl;
        for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query) // Iterates over all queries in allqueries
                {
                    auto ranking = ranker.score(*idx, *query, 50); // Returns a ranked list of the top 1000 documents for the current query
                    eval.avg_p(ranking,(*query).id(),50); // eval.avg_p stores the value of average precision for the current query in the instance eval
                }
                writeout <<  mu[i] << "\t" << eval.map() << "\n";

			std::cout << eval.map() << "\n" ;
                if (eval.map() > maxmap) // Updates maxmap, cmax, smax if the current map, which is equal to eval.map(), is greater than maxmap
                {
                    // You should only change the values of the following three assignments
                    maxmap = eval.map(); // Change 0 to the correct value DONE
                    mumax = mu[i];
                }

                eval.reset_stats(); // Deletes all the average precision values stored in eval to allow correct calculation of MAP for the next parameter combination
            }

    writeout.close();
    std::cout<<"Max MAP = "<< maxmap << " achieved by " << "mu = " << mumax << std::endl; // Prints to the standard ouput
    mu_ = mumax; // Returns the best s value to the calling function
}

int main(int argc, char* argv[]) {
      //index::register_ranker<new_ranker>();

  if (argc != 2 && argc != 3) {
    std::cerr << "Usage:\t" << argv[0] << " configFile" << std::endl;
    return 1;
  }

  // Log to standard error
  logging::set_cerr_logging();

  // Register additional analyzers
  parser::register_analyzers();
  sequence::register_analyzers();

  // Submission-specific - Ignore
  std::ofstream submission;

  submission.open("Assignment2/output.txt");
  if (!submission.is_open()) {
    std::cout << "Problem writing the output to the system. Make sure the "
                 "program has enough writing privileges. Quiting..."
              << std::endl;
    return 0;
  }
  std::string nickname;
  std::cout << "Enter your nickname: ";
  std::getline(std::cin, nickname);
  submission << nickname + '\n';
  // End of the submission-specific code

  //  Create an inverted index using a DBLRU cache.
  auto config = cpptoml::parse_file(argv[1]);
  auto idx = index::make_index<index::dblru_inverted_index>(*config, 30000);

  // Create a ranking class based on the config file.
  auto group = config->get_table("ranker");
  if (!group)
    throw std::runtime_error{"\"ranker\" group needed in config file!"};
  auto ranker = index::make_ranker(*group);

  // Get the path to the file containing queries
  auto query_path = config->get_as<std::string>("querypath");
  if (!query_path)
    throw std::runtime_error{"config file needs a \"querypath\" parameter"};

  std::ifstream queries{*query_path + *config->get_as<std::string>("dataset") +
                        "-queries.txt"};

  // Create an instance of ir_eval to evaluate the MAP and Precision@10 for the
  // training queries
  auto eval = index::ir_eval(*config);

  // Print the precision@10 and the MAP for the training queries
  size_t i = 1;
  std::string content;

 std::vector<corpus::document> trainqueries; // will contain train quesries (used for tuning function)

//std::vector<corpus::document> trainqueries; // will contain train quesries (used for tuning function)
  
while (i <= 70 && queries.good()) {
    std::getline(
        queries,
        content);  // Read the content of the current training query from file

    corpus::document query{
        doc_id{i - 1}};  // Instantiate the query as an empty document

    query.content(content);  // Set the content of the query

   trainqueries.push_back(query);
   std::cout << "Ranking query " << i++ << ": " << content << std::endl;

    auto ranking = ranker->score(
        *idx, query,
        50);  // ranking is a vector of pairs of the form <docID,docScore>
    // You can access the ith document's ID using ranking[i].d_id and its score
    // using ranking[i].score

    std::cout << "Precision@10 for this query: "
              << eval.precision(ranking, query.id(), 10) << std::endl;

    eval.avg_p(ranking, query.id(), 50);  // Store the average precision at 50
                                          // documents for the current query

    std::cout << "Showing top 10 of " << ranking.size() << " results."
              << std::endl;

    for (size_t i = 0; i < ranking.size() && i < 10;
         ++i)  // Loop over the top 10 documents in ranking
      std::cout << (i + 1) << ". "
                << " "
                << *idx->metadata(ranking[i].d_id).get<std::string>("name")
                << " " << ranking[i].score << std::endl;

    std::cout << std::endl;
  }

  std::cout << "The MAP for the training queries is: " << eval.map()
            << std::endl;

//  float k1 = 1.2;
//  float b = 0.75;
//  float k3 = 500.0;
//  bm25_tune (idx, trainqueries, eval, k1,b,k3);
    
// float l = 0.1;
// jm_tune(idx, trainqueries, eval, l); 

//float s_ = 0.1;
//pivot_tune(idx, trainqueries, eval, s_); 
    
//  float mu_ = 2000;
//  dirichlet_tune(idx, trainqueries, eval, mu_); 

    float alpha = 0.3;
    float lambda = 0.7;
   float s = 0.2;
    float mu = 500;
    int Q = 50;
    mptf2ln_tune(idx, trainqueries, eval, alpha, lambda, s, mu, Q); 

  // Write the top 50 documents of each test query to the submission file
  while (queries.good()) {
    std::getline(
        queries,
        content);  // Read the content of the current testing query from file

    corpus::document query{doc_id{i - 1}};

    query.content(content);

    auto ranking = ranker->score(*idx, query, 50);

    for (size_t i = 0; i < ranking.size() && i < 50;
         i++)  // Loop over the top 50 documents
    {
      submission
          << std::to_string(ranking[i].d_id) +
                 " ";  // Write the IDs of the top 50 documents to output.txt
    }
    submission << "\n";
  }

  submission.close();
  return 0;
}
