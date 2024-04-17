#include <eigen3/Eigen/Dense>

class GraphAttentionLayerOut {
public:
    GraphAttentionLayerOut(int in_features, int out_features, double dropout, double alpha, bool concat)
        : in_features(in_features), out_features(out_features), dropout(dropout), alpha(alpha), concat(concat) {
        // Initialisation des paramètres
        W_out.resize(in_features, out_features);
        a_out.resize(2 * out_features, 1);
        initializeParameters();
    }

    // Forward pass de la couche de sortie
    Eigen::MatrixXd forward(const Eigen::MatrixXd& h) {
        Eigen::MatrixXd Wh_out = h * W_out; // h.shape: (N, in_features), Wh_out.shape: (N, out_features)
        Eigen::MatrixXd e_out = prepareAttentionalMechanismInput(Wh_out);
        Eigen::MatrixXd zeroVec_out =-9e15 * Eigen::MatrixXd::Ones(e_out.rows(),e_out.cols());
        
        Eigen::MatrixXd attention_out = e_out.array() > 0 ? e_out : zeroVec_out.array();
        attention_out = (attention_out.array().exp() / attention_out.array().exp().sum(1)).matrix();
        attention_out = attention_out.unaryExpr([this](double x) { return dropNode(x); });

        Eigen::MatrixXd hPrime_out = attention_out * Wh_out;

        if (concat) {
            return hPrime_out.unaryExpr([this](double x) { return elu(x); });
        } else {
            return hPrime_out;
        }
    }

    // Backward pass de la couche de sortie
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& grad_output) {
        // Calcul des gradients pour la couche de sortie
        Eigen::MatrixXd grad_h_out = grad_output;
        Eigen::MatrixXd grad_Wh_out = attention_backward(grad_h_out);

        // Mise à jour des paramètres de la couche de sortie
        backward_update_parameters(h, grad_h_out);

        return std::make_tuple(grad_h_out, grad_Wh_out, W_out);
    }

    // Mise à jour des paramètres de la couche de sortie
    void backward_update_parameters(const Eigen::MatrixXd& h, const Eigen::MatrixXd& grad_h_out) {
        // Mise à jour des paramètres pour les poids W_out de la couche de sortie
        W_out = update_parameters(W_out, h, grad_h_out);
        // Mise à jour des paramètres pour les poids a_out de la couche de sortie
        a_out = update_parameters(a_out, h, grad_h_out);
    }

private:
    int in_features;
    int out_features;
    double dropout;
    double alpha;
    bool concat;
    Eigen::MatrixXd W_out;
    Eigen::MatrixXd a_out;

    void initializeParameters() {
        // Initialisation des paramètres pour les poids W_out de la couche de sortie
        W_out = Eigen::MatrixXd::Random(in_features, out_features);
        // Initialisation des paramètres pour les poids a_out de la couche de sortie
        a_out = Eigen::MatrixXd::Random(2 * out_features, 1);
    }

    Eigen::MatrixXd prepareAttentionalMechanismInput(const Eigen::MatrixXd& Wh_out) {
        // Calcul des entrées pour le mécanisme attentionnel de la couche de sortie
        Eigen::MatrixXd Wh1_out = Wh_out * a_out.topRows(out_features);
        Eigen::MatrixXd Wh2_out = Wh_out * a_out.bottomRows(out_features);
        return (Wh1_out.array() + Wh2_out.transpose().array()).matrix().unaryExpr([this](double x) { return leakyReLU(x); });
    }

    double leakyReLU(double x) {
        // Fonction d'activation leakyReLU
        return x > 0 ? x : alpha * x;
    }

    double dropNode(double x) {
        // Fonction de dropout pour la couche de sortie
        return (std::rand() / static_cast<double>(RAND_MAX)) > dropout ? x / (1 - dropout) : 0.0;
    }

    double elu(double x) {
        // Fonction d'activation elu
        return x > 0 ? x : alpha * (std::exp(x) - 1);
    }

    Eigen::MatrixXd attention_backward(const Eigen::MatrixXd& grad_h_out) {
        // Calcul des gradients pour le mécanisme attentionnel lors de la rétropropagation
        Eigen::MatrixXd grad_Wh_out = grad_h_out.array() * attention_derivative().array();
        return grad_Wh_out;
    }

    Eigen::MatrixXd attention_derivative() {
        // Calcul de la dérivée de la fonction d'attention
        return a_out.topRows(out_features).array() * a_out.bottomRows(out_features).transpose().array();
    }

    Eigen::MatrixXd update_parameters(const Eigen::MatrixXd& param, const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) {
        // Mise à jour des paramètres en utilisant une règle de mise à jour spécifique (par exemple, Adam)
        // Adapté en fonction de vos besoins spécifiques
        // Pseudo-code, vous devrez adapter cette fonction selon votre algorithme d'optimisation
        double learning_rate = 0.001; // Paramètre d'apprentissage
        double beta1 = 0.9; // Moment d'ordre 1
        double beta2 = 0.999; // Moment d'ordre 2
        double epsilon = 1e-8; // Petit terme pour éviter la division par zéro

        Eigen::MatrixXd grad = gradient.colwise().mean(); // Moyenne sur le batch
        Eigen::MatrixXd m = beta1 * param + (1 - beta1) * grad;
        Eigen::MatrixXd v = beta2 * param.array().square() + (1 - beta2) * grad.array().square();

        // Correction des biais
        double t = 1; // Numéro de l'itération
        Eigen::MatrixXd m_hat = m / (1 - std::pow(beta1, t));
        Eigen::MatrixXd v_hat = v / (1 - std::pow(beta2, t));

        // Mise à jour des paramètres
        Eigen::MatrixXd updated_param = param - learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon);

        return updated_param;
    }
};
