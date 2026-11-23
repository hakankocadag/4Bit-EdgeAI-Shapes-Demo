`timescale 1ns / 1ps
module top_accelerator #(
    parameter DATA_WIDTH = 4,
    parameter ACC_WIDTH = 24,
    parameter VECTOR_LENGTH = 784,
    parameter NUM_NEURONS = 3,
    parameter WEIGHT_FILE = "weights.hex",
    parameter ACT_FILE = "acts.hex",
    parameter BIAS_FILE = "biases.hex"       // Bias initialization file
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    output wire done,
    output wire output_valid,
    output wire signed [ACC_WIDTH-1:0] final_result,
    output wire [$clog2(NUM_NEURONS)-1:0] predicted_class
);
    localparam ACT_ADDR_WIDTH = $clog2(VECTOR_LENGTH) == 0 ? 1 : $clog2(VECTOR_LENGTH);
    localparam WEIGHT_ADDR_WIDTH = $clog2(VECTOR_LENGTH * NUM_NEURONS) == 0 ? 1 : $clog2(VECTOR_LENGTH * NUM_NEURONS);
    localparam NEURON_ADDR_WIDTH = $clog2(NUM_NEURONS) == 0 ? 1 : $clog2(NUM_NEURONS);

    wire mem_en, mac_en, acc_clr, agu_inc, agu_clr, relu_en;
    wire [ACT_ADDR_WIDTH-1:0] act_addr;
    wire [WEIGHT_ADDR_WIDTH-1:0] weight_addr;
    wire [DATA_WIDTH-1:0] weight_data, act_data;
    wire [ACC_WIDTH-1:0] mac_out;

    // Neuron counter from FSM (used for bias addressing)
    wire [NEURON_ADDR_WIDTH-1:0] neuron_count;

    control_fsm #(
        .VECTOR_LENGTH(VECTOR_LENGTH),
        .NUM_NEURONS(NUM_NEURONS),
        .ACT_ADDR_WIDTH(ACT_ADDR_WIDTH),
        .NEURON_ADDR_WIDTH(NEURON_ADDR_WIDTH)
    ) fsm_inst (
        .clk(clk), .rst_n(rst_n), .start(start),
        .current_act_index(act_addr),
        .mem_en(mem_en), .mac_en(mac_en), .acc_clr(acc_clr),
        .agu_inc(agu_inc), .agu_clr(agu_clr), .relu_en(relu_en),
        .output_valid(output_valid), .done(done),
        .neuron_count(neuron_count)  // Bias address index
    );

    agu #(
        .ACT_ADDR_WIDTH(ACT_ADDR_WIDTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .VECTOR_LENGTH(VECTOR_LENGTH)
    ) agu_inst (
        .clk(clk), .rst_n(rst_n), .agu_clr(agu_clr), .agu_inc(agu_inc),
        .act_addr(act_addr), .weight_addr(weight_addr)
    );

    memory_block #(
        .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .MEM_DEPTH(VECTOR_LENGTH * NUM_NEURONS), .INIT_FILE(WEIGHT_FILE)
    ) weight_mem (
        .clk(clk), .mem_en(mem_en), .addr(weight_addr), .read_data(weight_data)
    );

    memory_block #(
        .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ACT_ADDR_WIDTH),
        .MEM_DEPTH(VECTOR_LENGTH), .INIT_FILE(ACT_FILE)
    ) act_mem (
        .clk(clk), .mem_en(mem_en), .addr(act_addr), .read_data(act_data)
    );

    mac_unit #(.IN_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) mac_inst (
        .clk(clk), .rst_n(rst_n), .mac_en(mac_en), .acc_clr(acc_clr),
        .weight(weight_data), .activation(act_data), .mac_out(mac_out)
    );

    // =========================================================================
    // BIAS MEMORY + ADDER
    // Combinational read: bias value is ready immediately for current neuron.
    // Bias is added to each neuron's dot product before ReLU.
    // =========================================================================
    reg signed [ACC_WIDTH-1:0] bias_mem [0:NUM_NEURONS-1];
    initial begin
        $readmemh(BIAS_FILE, bias_mem);
    end

    // Read bias for the active neuron counter (combinational)
    wire signed [ACC_WIDTH-1:0] current_bias = bias_mem[neuron_count];

    // Add bias to MAC output: z = (W * x) + b
    wire signed [ACC_WIDTH-1:0] mac_plus_bias = $signed(mac_out) + current_bias;

    // ReLU operates on the biased pre-activation sum
    relu_unit #(.WIDTH(ACC_WIDTH)) relu_inst (
        .clk(clk), .rst_n(rst_n), .relu_en(relu_en),
        .data_in(mac_plus_bias),
        .data_out(final_result)
    );

    argmax_unit #(
        .SCORE_WIDTH(ACC_WIDTH),
        .NEURON_BITS(NEURON_ADDR_WIDTH)
    ) argmax_inst (
        .clk(clk), .rst_n(rst_n),
        .score_valid(output_valid),
        .score(final_result),
        .done_in(done),
        .predicted_class(predicted_class)
    );

endmodule
