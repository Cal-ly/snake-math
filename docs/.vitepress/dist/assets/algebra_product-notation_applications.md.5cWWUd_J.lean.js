import{_ as s,P as l,a as p,b as c,k as o,Q as a,d as n,M as r,j as i}from"./chunks/components.D7iw6peK.js";const h=JSON.parse('{"title":"Product Notation: Real-World Applications","description":"Discover how product notation powers machine learning, finance, probability, and quality control with practical PyScript examples.","frontmatter":{"title":"Product Notation: Real-World Applications","description":"Discover how product notation powers machine learning, finance, probability, and quality control with practical PyScript examples."},"headers":[],"relativePath":"algebra/product-notation/applications.md","filePath":"algebra/product-notation/applications.md"}'),u={name:"algebra/product-notation/applications.md"};function d(m,e,_,f,y,b){const t=l("py-script");return c(),p("div",null,[e[8]||(e[8]=o("",8)),a(t,null,{default:r(()=>e[0]||(e[0]=[i(" import math from collections import defaultdict "),n("p",null,'def naive_bayes_demo(): """Demonstrate Naive Bayes classification using product notation"""',-1),n("pre",null,[n("code",null,`print("Naive Bayes Text Classification Demo")
print("=" * 40)

# Training data: [text, label]
training_data = [
    ("love this movie amazing", "positive"),
    ("great film excellent acting", "positive"),
    ("wonderful story beautiful", "positive"),
    ("terrible movie boring", "negative"),
    ("awful film bad acting", "negative"),
    ("horrible story waste time", "negative"),
    ("good movie but slow", "neutral"),
    ("okay film average", "neutral"),
]

def train_naive_bayes(data):
    """Train Naive Bayes classifier"""
    
    # Count classes and words
    class_counts = defaultdict(int)
    word_counts = defaultdict(lambda: defaultdict(int))
    vocabulary = set()
    
    for text, label in data:
        class_counts[label] += 1
        words = text.split()
        for word in words:
            word_counts[label][word] += 1
            vocabulary.add(word)
    
    # Calculate probabilities
    total_docs = len(data)
    class_probs = {cls: count/total_docs for cls, count in class_counts.items()}
    
    # Word probabilities with Laplace smoothing
    word_probs = defaultdict(lambda: defaultdict(float))
    for cls in class_counts:
        total_words = sum(word_counts[cls].values())
        vocab_size = len(vocabulary)
        
        for word in vocabulary:
            # P(word|class) with Laplace smoothing
            word_probs[cls][word] = (word_counts[cls][word] + 1) / (total_words + vocab_size)
    
    return class_probs, word_probs, vocabulary

def classify_text(text, class_probs, word_probs, vocabulary):
    """Classify text using trained model"""
    
    words = text.split()
    scores = {}
    
    for cls in class_probs:
        # Start with class prior: P(class)
        log_prob = math.log(class_probs[cls])
        
        # Add log probabilities: log(∏ P(word|class)) = Σ log(P(word|class))
        for word in words:
            if word in vocabulary:
                log_prob += math.log(word_probs[cls][word])
        
        scores[cls] = log_prob
    
    # Return class with highest score
    predicted_class = max(scores, key=scores.get)
    return predicted_class, scores

# Train the model
class_probs, word_probs, vocabulary = train_naive_bayes(training_data)

# Display training results
print("Training Results:")
print(f"Classes: {list(class_probs.keys())}")
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Class probabilities: {dict(class_probs)}")

# Test classification
test_texts = [
    "amazing wonderful movie",
    "terrible awful film",
    "okay average story",
]

print(f"\\nClassification Results:")
for text in test_texts:
    predicted, scores = classify_text(text, class_probs, word_probs, vocabulary)
    print(f"Text: '{text}'")
    print(f"Predicted: {predicted}")
    print(f"Scores: {dict(scores)}")
    print()

return class_probs, word_probs
`)],-1),n("h1",{id:"run-the-demo",tabindex:"-1"},[i("Run the demo "),n("a",{class:"header-anchor",href:"#run-the-demo","aria-label":'Permalink to "Run the demo"'},"​")],-1),n("p",null,"naive_bayes_demo()",-1)])),_:1,__:[0]}),e[9]||(e[9]=n("h3",{id:"neural-network-gradients",tabindex:"-1"},[i("Neural Network Gradients "),n("a",{class:"header-anchor",href:"#neural-network-gradients","aria-label":'Permalink to "Neural Network Gradients"'},"​")],-1)),e[10]||(e[10]=n("p",null,"Product notation appears in backpropagation through the chain rule:",-1)),a(t,null,{default:r(()=>e[1]||(e[1]=[i(' def neural_network_gradients(): """Demonstrate gradient calculation using product notation""" '),n("pre",null,[n("code",null,`print("Neural Network Gradient Calculation")
print("=" * 40)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

class SimpleNeuron:
    """Simple neuron for gradient demonstration"""
    
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        """Forward pass: z = Σ(wᵢxᵢ) + b, a = σ(z)"""
        self.inputs = inputs
        self.z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.activation = sigmoid(self.z)
        return self.activation
    
    def backward(self, upstream_gradient):
        """Backward pass using chain rule products"""
        
        # Chain rule: ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
        
        # ∂a/∂z = σ'(z)
        da_dz = sigmoid_derivative(self.z)
        
        # ∂L/∂z = ∂L/∂a × ∂a/∂z
        dL_dz = upstream_gradient * da_dz
        
        # ∂z/∂w = x, so ∂L/∂w = ∂L/∂z × x
        weight_gradients = [dL_dz * x for x in self.inputs]
        
        # ∂z/∂b = 1, so ∂L/∂b = ∂L/∂z
        bias_gradient = dL_dz
        
        # ∂z/∂x = w, so ∂L/∂x = ∂L/∂z × w (for next layer)
        input_gradients = [dL_dz * w for w in self.weights]
        
        return weight_gradients, bias_gradient, input_gradients

# Example: 2-layer network
print("Example: 2-Layer Neural Network")

# Layer 1: 2 inputs → 3 neurons
layer1_neurons = [
    SimpleNeuron([0.5, 0.3], 0.1),
    SimpleNeuron([0.2, 0.7], -0.2),
    SimpleNeuron([0.6, 0.4], 0.0)
]

# Layer 2: 3 inputs → 1 output
layer2_neuron = SimpleNeuron([0.8, 0.5, 0.3], 0.1)

# Forward pass
inputs = [1.0, 0.5]

# Layer 1 forward
layer1_outputs = []
for neuron in layer1_neurons:
    output = neuron.forward(inputs)
    layer1_outputs.append(output)

# Layer 2 forward
final_output = layer2_neuron.forward(layer1_outputs)

print(f"Input: {inputs}")
print(f"Layer 1 outputs: {[f'{x:.4f}' for x in layer1_outputs]}")
print(f"Final output: {final_output:.4f}")

# Backward pass (assuming target = 0.8)
target = 0.8
loss = 0.5 * (final_output - target) ** 2

# Gradient of loss w.r.t. output
dL_da = final_output - target

print(f"\\nBackward Pass:")
print(f"Loss: {loss:.4f}")
print(f"Output gradient: {dL_da:.4f}")

# Layer 2 backward
l2_weight_grads, l2_bias_grad, l2_input_grads = layer2_neuron.backward(dL_da)

print(f"Layer 2 weight gradients: {[f'{g:.4f}' for g in l2_weight_grads]}")
print(f"Layer 2 bias gradient: {l2_bias_grad:.4f}")

# Layer 1 backward
for i, neuron in enumerate(layer1_neurons):
    w_grads, b_grad, _ = neuron.backward(l2_input_grads[i])
    print(f"Layer 1 neuron {i+1} weight gradients: {[f'{g:.4f}' for g in w_grads]}")

print(f"\\nKey Insight: Each gradient calculation uses the chain rule")
print(f"which creates products of partial derivatives along the path")
print(f"from loss back to each parameter.")

return loss, l2_weight_grads
`)],-1),n("p",null,"neural_network_gradients()",-1)])),_:1,__:[1]}),e[11]||(e[11]=n("h2",{id:"financial-modeling",tabindex:"-1"},[i("Financial Modeling "),n("a",{class:"header-anchor",href:"#financial-modeling","aria-label":'Permalink to "Financial Modeling"'},"​")],-1)),e[12]||(e[12]=n("h3",{id:"compound-interest-and-investment-growth",tabindex:"-1"},[i("Compound Interest and Investment Growth "),n("a",{class:"header-anchor",href:"#compound-interest-and-investment-growth","aria-label":'Permalink to "Compound Interest and Investment Growth"'},"​")],-1)),e[13]||(e[13]=n("p",null,"Product notation naturally models compound growth over time:",-1)),a(t,null,{default:r(()=>e[2]||(e[2]=[i(' def compound_interest_analysis(): """Analyze compound interest using product notation""" '),n("pre",null,[n("code",null,`print("Compound Interest Analysis")
print("=" * 30)

def compound_growth(principal, rates):
    """Calculate compound growth: A = P × ∏(1 + rᵢ)"""
    growth_factors = [1 + rate for rate in rates]
    product = math.prod(growth_factors)
    final_value = principal * product
    return final_value, growth_factors

def equivalent_annual_rate(rates):
    """Calculate equivalent annual rate from multiple periods"""
    # (1 + r_eq)^n = ∏(1 + r_i)
    product_growth = math.prod(1 + rate for rate in rates)
    n_periods = len(rates)
    equivalent_rate = product_growth ** (1/n_periods) - 1
    return equivalent_rate

# Example: Variable interest rates over 5 years
principal = 10000
annual_rates = [0.03, 0.045, 0.02, 0.055, 0.04]  # 3%, 4.5%, 2%, 5.5%, 4%

final_value, growth_factors = compound_growth(principal, annual_rates)
equiv_rate = equivalent_annual_rate(annual_rates)

print(f"Principal: \${principal:,}")
print(f"Annual rates: {[f'{r:.1%}' for r in annual_rates]}")
print(f"Growth factors: {[f'{g:.4f}' for g in growth_factors]}")
print(f"Product of growth factors: {math.prod(growth_factors):.4f}")
print(f"Final value: \${final_value:,.2f}")
print(f"Equivalent annual rate: {equiv_rate:.2%}")

# Verification with equivalent rate
verification = principal * (1 + equiv_rate) ** len(annual_rates)
print(f"Verification with equivalent rate: \${verification:,.2f}")
print(f"Match: {abs(final_value - verification) < 0.01}")

return final_value, equiv_rate
`)],-1),n("p",null,"compound_interest_analysis()",-1)])),_:1,__:[2]}),e[14]||(e[14]=n("h3",{id:"portfolio-performance-analysis",tabindex:"-1"},[i("Portfolio Performance Analysis "),n("a",{class:"header-anchor",href:"#portfolio-performance-analysis","aria-label":'Permalink to "Portfolio Performance Analysis"'},"​")],-1)),a(t,null,{default:r(()=>e[3]||(e[3]=[i(' def portfolio_performance(): """Analyze portfolio performance using geometric returns""" '),n("pre",null,[n("code",null,`print("\\nPortfolio Performance Analysis")
print("=" * 30)

def geometric_return(returns):
    """Geometric return = ∏(1 + r_i) - 1"""
    growth_product = math.prod(1 + r for r in returns)
    return growth_product - 1

def annualized_return(returns, periods_per_year=12):
    """Annualized return from periodic returns"""
    total_periods = len(returns)
    growth_product = math.prod(1 + r for r in returns)
    years = total_periods / periods_per_year
    annualized = growth_product ** (1/years) - 1
    return annualized

def volatility_analysis(returns):
    """Calculate volatility and performance metrics"""
    geometric_ret = geometric_return(returns)
    arithmetic_mean = sum(returns) / len(returns)
    
    # Standard deviation of returns
    variance = sum((r - arithmetic_mean)**2 for r in returns) / (len(returns) - 1)
    volatility = math.sqrt(variance)
    
    return geometric_ret, arithmetic_mean, volatility

# Example: Monthly portfolio returns
monthly_returns = [
    0.02, -0.01, 0.03, 0.015, -0.005, 0.025,
    0.01, 0.035, -0.02, 0.008, 0.012, 0.018
]

print(f"Monthly returns: {[f'{r:.1%}' for r in monthly_returns]}")

geometric_ret = geometric_return(monthly_returns)
annualized_ret = annualized_return(monthly_returns, 12)
geom_ret, arith_mean, vol = volatility_analysis(monthly_returns)

print(f"\\nPerformance Metrics:")
print(f"Geometric return (total): {geometric_ret:.2%}")
print(f"Annualized return: {annualized_ret:.2%}")
print(f"Arithmetic mean (monthly): {arith_mean:.2%}")
print(f"Monthly volatility: {vol:.2%}")
print(f"Annualized volatility: {vol * math.sqrt(12):.2%}")

# Compare arithmetic vs geometric returns
arithmetic_total = sum(monthly_returns)
print(f"\\nComparison:")
print(f"Arithmetic sum: {arithmetic_total:.2%}")
print(f"Geometric return: {geometric_ret:.2%}")
print(f"Difference: {geometric_ret - arithmetic_total:.2%}")

return geometric_ret, annualized_ret, vol
`)],-1),n("p",null,"portfolio_performance()",-1)])),_:1,__:[3]}),e[15]||(e[15]=n("h2",{id:"quality-control-and-manufacturing",tabindex:"-1"},[i("Quality Control and Manufacturing "),n("a",{class:"header-anchor",href:"#quality-control-and-manufacturing","aria-label":'Permalink to "Quality Control and Manufacturing"'},"​")],-1)),e[16]||(e[16]=n("h3",{id:"reliability-engineering",tabindex:"-1"},[i("Reliability Engineering "),n("a",{class:"header-anchor",href:"#reliability-engineering","aria-label":'Permalink to "Reliability Engineering"'},"​")],-1)),e[17]||(e[17]=n("p",null,"Product notation models system reliability when components operate independently:",-1)),a(t,null,{default:r(()=>e[4]||(e[4]=[i(' def reliability_analysis(): """Analyze system reliability using product notation""" '),n("pre",null,[n("code",null,`print("\\nReliability Engineering Analysis")
print("=" * 35)

def system_reliability(component_reliabilities, configuration="series"):
    """Calculate system reliability based on configuration"""
    
    if configuration == "series":
        # Series: R_system = ∏ R_i (all must work)
        return math.prod(component_reliabilities)
    
    elif configuration == "parallel":
        # Parallel: R_system = 1 - ∏(1 - R_i) (at least one must work)
        failure_probs = [1 - r for r in component_reliabilities]
        system_failure = math.prod(failure_probs)
        return 1 - system_failure
    
    elif configuration == "k_out_of_n":
        # k-out-of-n systems (simplified for equal components)
        # More complex calculation, simplified here
        n = len(component_reliabilities)
        r = component_reliabilities[0]  # Assume equal reliabilities
        
        from math import comb
        reliability = 0
        for i in range(2, n+1):  # Assuming k=2 (at least 2 must work)
            reliability += comb(n, i) * (r**i) * ((1-r)**(n-i))
        return reliability

def mtbf_analysis(failure_rates):
    """Mean Time Between Failure analysis"""
    
    # For series system: λ_system = Σ λ_i
    system_failure_rate = sum(failure_rates)
    system_mtbf = 1 / system_failure_rate
    
    # Individual MTBFs
    individual_mtbfs = [1/rate for rate in failure_rates]
    
    return system_mtbf, individual_mtbfs, system_failure_rate

# Example: 5-component electronic system
component_reliabilities = [0.95, 0.98, 0.92, 0.97, 0.94]
component_names = ["Power Supply", "CPU", "Memory", "Storage", "Network"]

print("Component Reliabilities:")
for name, rel in zip(component_names, component_reliabilities):
    print(f"  {name}: {rel:.1%}")

# Calculate different configurations
series_reliability = system_reliability(component_reliabilities, "series")
parallel_reliability = system_reliability(component_reliabilities, "parallel")

print(f"\\nSystem Reliability:")
print(f"Series configuration: {series_reliability:.3%}")
print(f"Parallel configuration: {parallel_reliability:.3%}")

# MTBF Analysis (failure rates in failures per hour)
failure_rates = [0.001, 0.0005, 0.002, 0.0008, 0.0015]  # per hour

system_mtbf, individual_mtbfs, sys_fail_rate = mtbf_analysis(failure_rates)

print(f"\\nMTBF Analysis:")
print(f"System failure rate: {sys_fail_rate:.4f} failures/hour")
print(f"System MTBF: {system_mtbf:.0f} hours ({system_mtbf/24:.1f} days)")

# Component contribution analysis
print(f"\\nComponent MTBF Contributions:")
for name, mtbf, rate in zip(component_names, individual_mtbfs, failure_rates):
    contribution = rate / sys_fail_rate * 100
    print(f"  {name}: {mtbf:.0f}h (contributes {contribution:.1f}% to system failure)")

return series_reliability, parallel_reliability, system_mtbf
`)],-1),n("p",null,"reliability_analysis()",-1)])),_:1,__:[4]}),e[18]||(e[18]=n("h3",{id:"quality-control-sampling",tabindex:"-1"},[i("Quality Control Sampling "),n("a",{class:"header-anchor",href:"#quality-control-sampling","aria-label":'Permalink to "Quality Control Sampling"'},"​")],-1)),a(t,null,{default:r(()=>e[5]||(e[5]=[i(' def quality_control_sampling(): """Quality control using product notation for batch acceptance""" '),n("pre",null,[n("code",null,`print("\\nQuality Control Sampling")
print("=" * 28)

def acceptance_probability(defect_rate, sample_size, max_defects=0):
    """Calculate probability of accepting a batch"""
    
    # Probability that ALL sampled items are good
    # P(accept) = P(≤ max_defects in sample)
    
    if max_defects == 0:
        # P(all good) = (1 - p)^n
        good_rate = 1 - defect_rate
        return good_rate ** sample_size
    else:
        # Binomial calculation for more complex cases
        from math import comb
        prob = 0
        for k in range(max_defects + 1):
            prob += comb(sample_size, k) * (defect_rate**k) * ((1-defect_rate)**(sample_size-k))
        return prob

def producer_risk_analysis(true_quality, sample_sizes, max_defects=0):
    """Analyze producer's risk (rejecting good batches)"""
    
    risks = []
    for n in sample_sizes:
        accept_prob = acceptance_probability(true_quality, n, max_defects)
        reject_prob = 1 - accept_prob  # Producer's risk
        risks.append(reject_prob)
    
    return risks

def consumer_risk_analysis(poor_quality, sample_sizes, max_defects=0):
    """Analyze consumer's risk (accepting poor batches)"""
    
    risks = []
    for n in sample_sizes:
        accept_prob = acceptance_probability(poor_quality, n, max_defects)
        risks.append(accept_prob)  # Consumer's risk
    
    return risks

# Example: Electronics manufacturing
print("Electronics Manufacturing Quality Control")

# Scenario parameters
good_quality = 0.01    # 1% defect rate (acceptable)
poor_quality = 0.10    # 10% defect rate (unacceptable)
sample_sizes = [5, 10, 20, 50, 100]

print(f"Good quality level: {good_quality:.1%} defects")
print(f"Poor quality level: {poor_quality:.1%} defects")
print(f"Acceptance criterion: 0 defects in sample")

# Calculate risks
producer_risks = producer_risk_analysis(good_quality, sample_sizes)
consumer_risks = consumer_risk_analysis(poor_quality, sample_sizes)

print(f"\\nSampling Plan Analysis:")
print("Sample Size | Producer Risk | Consumer Risk | Accept Good | Accept Poor")
print("-" * 70)

for n, prod_risk, cons_risk in zip(sample_sizes, producer_risks, consumer_risks):
    accept_good = 1 - prod_risk
    accept_poor = cons_risk
    print(f"{n:^11} | {prod_risk:^13.1%} | {cons_risk:^13.1%} | {accept_good:^11.1%} | {accept_poor:^11.1%}")

# Recommended sample size analysis
print(f"\\nRecommendations:")
for i, n in enumerate(sample_sizes):
    if producer_risks[i] < 0.05 and consumer_risks[i] < 0.10:
        print(f"Sample size {n} meets criteria (Producer risk < 5%, Consumer risk < 10%)")
        break
else:
    print("Consider larger sample sizes or different acceptance criteria")

# Batch size impact
print(f"\\nBatch Processing:")
batch_size = 1000
daily_batches = 10

# Expected daily rejections
daily_good_batches = 8  # Most batches are good quality
daily_poor_batches = 2  # Some batches are poor quality

sample_size = 20
good_accept_rate = 1 - producer_risk_analysis(good_quality, [sample_size])[0]
poor_accept_rate = consumer_risk_analysis(poor_quality, [sample_size])[0]

expected_good_accepted = daily_good_batches * good_accept_rate
expected_poor_accepted = daily_poor_batches * poor_accept_rate

print(f"Daily processing with sample size {sample_size}:")
print(f"Good batches accepted: {expected_good_accepted:.1f}/{daily_good_batches}")
print(f"Poor batches wrongly accepted: {expected_poor_accepted:.1f}/{daily_poor_batches}")

return producer_risks, consumer_risks
`)],-1),n("p",null,"quality_control_sampling()",-1)])),_:1,__:[5]}),e[19]||(e[19]=n("h2",{id:"combinatorics-and-probability",tabindex:"-1"},[i("Combinatorics and Probability "),n("a",{class:"header-anchor",href:"#combinatorics-and-probability","aria-label":'Permalink to "Combinatorics and Probability"'},"​")],-1)),e[20]||(e[20]=n("h3",{id:"complex-event-probabilities",tabindex:"-1"},[i("Complex Event Probabilities "),n("a",{class:"header-anchor",href:"#complex-event-probabilities","aria-label":'Permalink to "Complex Event Probabilities"'},"​")],-1)),a(t,null,{default:r(()=>e[6]||(e[6]=[i(' def complex_probability_calculations(): """Complex probability problems using product notation""" '),n("pre",null,[n("code",null,`print("\\nComplex Probability Calculations")
print("=" * 35)

def tournament_bracket_probability():
    """Calculate probability of specific tournament outcomes"""
    
    print("Tournament Bracket Analysis:")
    
    # Example: 8-team single elimination tournament
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    # Win probabilities for each matchup (team vs next team)
    # Round 1: A vs B, C vs D, E vs F, G vs H
    round1_probs = [0.6, 0.7, 0.55, 0.8]  # P(A beats B), P(C beats D), etc.
    
    # Probability that teams A, C, E, G all advance to semifinals
    prob_desired_semifinal = math.prod(round1_probs)
    
    print(f"Teams: {teams}")
    print(f"Round 1 win probabilities: {round1_probs}")
    print(f"Probability of A, C, E, G reaching semifinals: {prob_desired_semifinal:.3f}")
    
    # Continue to finals
    # Semifinals: A vs C, E vs G
    semifinal_probs = [0.65, 0.7]  # P(A beats C), P(E beats G)
    
    # Finals: A vs E
    final_prob = 0.55  # P(A beats E)
    
    # Probability A wins the entire tournament via this path
    full_path_prob = math.prod(round1_probs[:1] + [round1_probs[1]] + 
                             round1_probs[2:] + semifinal_probs[:1] + [final_prob])
    
    print(f"Probability A wins tournament via specific path: {full_path_prob:.4f}")
    
    return prob_desired_semifinal, full_path_prob

def quality_assurance_chain():
    """Multi-stage quality assurance process"""
    
    print(f"\\nMulti-Stage Quality Assurance:")
    
    # Production stages with pass rates
    stages = ["Assembly", "Testing", "Calibration", "Final QC", "Packaging"]
    pass_rates = [0.95, 0.98, 0.97, 0.99, 0.995]
    
    # Overall yield = ∏ pass_rates
    overall_yield = math.prod(pass_rates)
    
    print(f"Production stages and pass rates:")
    for stage, rate in zip(stages, pass_rates):
        print(f"  {stage}: {rate:.1%}")
    
    print(f"Overall yield: {overall_yield:.2%}")
    
    # Cost analysis
    unit_cost_per_stage = [10, 5, 8, 3, 2]  # Cumulative costs
    cumulative_costs = []
    cumulative_yield = 1.0
    
    print(f"\\nCumulative Analysis:")
    total_cost = 0
    for i, (stage, rate, cost) in enumerate(zip(stages, pass_rates, unit_cost_per_stage)):
        cumulative_yield *= rate
        total_cost += cost
        cumulative_costs.append(total_cost)
        
        print(f"Through {stage}: {cumulative_yield:.2%} yield, \${total_cost} cost")
    
    # Expected cost per good unit
    expected_cost_per_good = total_cost / overall_yield
    print(f"Expected cost per good unit: \${expected_cost_per_good:.2f}")
    
    return overall_yield, expected_cost_per_good

def network_reliability():
    """Network path reliability analysis"""
    
    print(f"\\nNetwork Reliability Analysis:")
    
    # Network with multiple paths between source and destination
    # Path 1: A → B → C (3 links)
    # Path 2: A → D → E → C (4 links)
    # Path 3: A → F → C (2 links)
    
    # Link reliabilities
    link_reliabilities = {
        'AB': 0.95, 'BC': 0.97, 'AD': 0.93, 'DE': 0.96, 
        'EC': 0.94, 'AF': 0.99, 'FC': 0.92
    }
    
    # Path reliabilities (product of link reliabilities)
    path1_reliability = link_reliabilities['AB'] * link_reliabilities['BC']
    path2_reliability = link_reliabilities['AD'] * link_reliabilities['DE'] * link_reliabilities['EC']
    path3_reliability = link_reliabilities['AF'] * link_reliabilities['FC']
    
    print(f"Path reliabilities:")
    print(f"  Path 1 (A→B→C): {path1_reliability:.3f}")
    print(f"  Path 2 (A→D→E→C): {path2_reliability:.3f}")
    print(f"  Path 3 (A→F→C): {path3_reliability:.3f}")
    
    # Overall network reliability (at least one path works)
    # R = 1 - ∏(1 - R_path_i)
    path_failure_probs = [1 - path1_reliability, 1 - path2_reliability, 1 - path3_reliability]
    network_failure = math.prod(path_failure_probs)
    network_reliability = 1 - network_failure
    
    print(f"Network reliability: {network_reliability:.4f}")
    
    # Importance of each path
    print(f"\\nPath Importance Analysis:")
    for i, (path_num, path_rel) in enumerate([(1, path1_reliability), (2, path2_reliability), (3, path3_reliability)]):
        # Network reliability without this path
        other_paths = [path1_reliability, path2_reliability, path3_reliability]
        other_paths.pop(i)
        other_failure_probs = [1 - r for r in other_paths]
        network_without_path = 1 - math.prod(other_failure_probs)
        
        importance = network_reliability - network_without_path
        print(f"  Path {path_num} importance: {importance:.4f}")
    
    return network_reliability, path1_reliability, path2_reliability, path3_reliability

# Run all analyses
tournament_result = tournament_bracket_probability()
qa_result = quality_assurance_chain()
network_result = network_reliability()

print(f"\\nSummary:")
print(f"• Product notation enables complex probability calculations")
print(f"• Independent events multiply their probabilities")
print(f"• Series systems require ALL components to work")
print(f"• Parallel systems need only ONE path to work")

return tournament_result, qa_result, network_result
`)],-1),n("p",null,"complex_probability_calculations()",-1)])),_:1,__:[6]}),e[21]||(e[21]=n("h2",{id:"performance-and-optimization",tabindex:"-1"},[i("Performance and Optimization "),n("a",{class:"header-anchor",href:"#performance-and-optimization","aria-label":'Permalink to "Performance and Optimization"'},"​")],-1)),e[22]||(e[22]=n("h3",{id:"algorithmic-complexity",tabindex:"-1"},[i("Algorithmic Complexity "),n("a",{class:"header-anchor",href:"#algorithmic-complexity","aria-label":'Permalink to "Algorithmic Complexity"'},"​")],-1)),e[23]||(e[23]=n("p",null,"Product notation appears in complexity analysis, especially for nested algorithms:",-1)),a(t,null,{default:r(()=>e[7]||(e[7]=[i(' def algorithmic_complexity_analysis(): """Analyze algorithmic complexity using product notation""" '),n("pre",null,[n("code",null,`print("\\nAlgorithmic Complexity Analysis")
print("=" * 35)

def matrix_multiplication_complexity():
    """Analyze matrix multiplication complexity"""
    
    print("Matrix Multiplication Complexity:")
    
    # Standard algorithm: O(∏ᵢ dᵢ) for chain multiplication
    # A₁(d₀×d₁) × A₂(d₁×d₂) × ... × Aₙ(dₙ₋₁×dₙ)
    
    dimensions = [100, 50, 200, 75, 300]  # Matrix dimensions
    n_matrices = len(dimensions) - 1
    
    print(f"Matrix chain dimensions: {dimensions}")
    print(f"Number of matrices: {n_matrices}")
    
    # Calculate operations for different association orders
    def chain_multiply_ops(dims, i, j):
        """Calculate operations for multiplying matrices i through j"""
        if i == j:
            return 0
        
        min_ops = float('inf')
        for k in range(i, j):
            left_ops = chain_multiply_ops(dims, i, k)
            right_ops = chain_multiply_ops(dims, k+1, j)
            merge_ops = dims[i] * dims[k+1] * dims[j+1]
            total_ops = left_ops + right_ops + merge_ops
            min_ops = min(min_ops, total_ops)
        
        return min_ops
    
    # Left-to-right association
    left_to_right_ops = 0
    current_dims = [dimensions[0], dimensions[1]]
    
    for i in range(1, n_matrices):
        # Multiply current result with next matrix
        ops = current_dims[0] * current_dims[1] * dimensions[i+1]
        left_to_right_ops += ops
        current_dims[1] = dimensions[i+1]
    
    # Optimal association (simplified calculation)
    optimal_ops = chain_multiply_ops(dimensions, 0, n_matrices-1)
    
    print(f"Left-to-right operations: {left_to_right_ops:,}")
    print(f"Optimal operations: {optimal_ops:,}")
    print(f"Improvement factor: {left_to_right_ops / optimal_ops:.2f}x")
    
    return left_to_right_ops, optimal_ops

def nested_loop_analysis():
    """Analyze nested loop complexity"""
    
    print(f"\\nNested Loop Complexity:")
    
    # Example: Multi-dimensional array processing
    array_dimensions = [50, 30, 20, 15]  # 4D array
    
    # Total operations = ∏ᵢ dᵢ
    total_operations = math.prod(array_dimensions)
    
    print(f"Array dimensions: {array_dimensions}")
    print(f"Total operations: {total_operations:,}")
    
    # Memory access patterns
    def analyze_access_pattern(dims, pattern="row_major"):
        """Analyze memory access efficiency"""
        
        total_elements = math.prod(dims)
        
        if pattern == "row_major":
            # Contiguous access in innermost dimension
            cache_misses = total_elements // dims[-1]  # Simplified
        else:  # column_major
            # Non-contiguous access
            cache_misses = total_elements // dims[0]  # Simplified
        
        return cache_misses
    
    row_major_misses = analyze_access_pattern(array_dimensions, "row_major")
    col_major_misses = analyze_access_pattern(array_dimensions, "column_major")
    
    print(f"Estimated cache misses (row-major): {row_major_misses:,}")
    print(f"Estimated cache misses (column-major): {col_major_misses:,}")
    print(f"Row-major advantage: {col_major_misses / row_major_misses:.1f}x fewer misses")
    
    return total_operations, row_major_misses, col_major_misses

def parallel_processing_efficiency():
    """Analyze parallel processing efficiency"""
    
    print(f"\\nParallel Processing Efficiency:")
    
    # Problem: Process N×M×K data cube
    problem_size = [1000, 800, 600]
    total_work = math.prod(problem_size)
    
    # Different parallelization strategies
    processors = [1, 2, 4, 8, 16, 32]
    
    print(f"Problem size: {problem_size}")
    print(f"Total work units: {total_work:,}")
    
    def parallel_efficiency(work, p_count, overhead_factor=0.1):
        """Calculate parallel efficiency"""
        
        ideal_speedup = p_count
        work_per_processor = work / p_count
        
        # Communication overhead grows with processor count
        overhead = overhead_factor * math.log2(p_count) if p_count > 1 else 0
        
        actual_time = work_per_processor + overhead * work_per_processor
        serial_time = work
        
        actual_speedup = serial_time / (actual_time * p_count)
        efficiency = actual_speedup / ideal_speedup
        
        return actual_speedup, efficiency
    
    print(f"\\nParallel Performance Analysis:")
    print("Processors | Speedup | Efficiency | Work/Proc")
    print("-" * 45)
    
    for p in processors:
        speedup, efficiency = parallel_efficiency(total_work, p)
        work_per_proc = total_work / p
        print(f"{p:^10} | {speedup:^7.1f} | {efficiency:^10.1%} | {work_per_proc:^9.0f}")
    
    return total_work, speedup, efficiency

# Run all analyses
matrix_result = matrix_multiplication_complexity()
loop_result = nested_loop_analysis()
parallel_result = parallel_processing_efficiency()

print(f"\\nComplexity Analysis Summary:")
print(f"• Product notation captures multiplicative growth in algorithms")
print(f"• Matrix chain multiplication shows importance of association order")
print(f"• Nested loops create complexity proportional to dimension products")
print(f"• Parallel efficiency depends on work distribution and communication")

return matrix_result, loop_result, parallel_result
`)],-1),n("p",null,"algorithmic_complexity_analysis()",-1)])),_:1,__:[7]}),e[24]||(e[24]=o("",17))])}const w=s(u,[["render",d]]);export{h as __pageData,w as default};
