`timescale 1ns / 1ps
// =============================================================================
//  Control FSM -- 6-State One-Hot Moore Inference Controller
// =============================================================================
module control_fsm #(
    parameter VECTOR_LENGTH = 784,
    parameter NUM_NEURONS = 3,
    parameter ACT_ADDR_WIDTH = 10,
    parameter NEURON_ADDR_WIDTH = 2
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [ACT_ADDR_WIDTH-1:0] current_act_index,
    output reg mem_en,
    output reg mac_en,
    output reg acc_clr,
    output reg agu_inc,
    output reg agu_clr,
    output reg relu_en,
    output reg output_valid,
    output reg done,
    output wire [NEURON_ADDR_WIDTH-1:0] neuron_count  // Neuron counter for bias addressing
);
    localparam S_IDLE       = 6'b000001;
    localparam S_FETCH      = 6'b000010;
    localparam S_MAC        = 6'b000100;
    localparam S_RELU       = 6'b001000;
    localparam S_OUTPUT     = 6'b010000;
    localparam S_DONE       = 6'b100000;

    reg [5:0] current_state, next_state;
    reg [NEURON_ADDR_WIDTH-1:0] internal_neuron_count;

    // Expose internal_neuron_count as the bias memory address
    assign neuron_count = internal_neuron_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) current_state <= S_IDLE;
        else current_state <= next_state;
    end

    always @(*) begin
        next_state = current_state;
        case (current_state)
            S_IDLE: if (start) next_state = S_FETCH;
            S_FETCH: next_state = S_MAC;
            S_MAC: begin
                if (current_act_index == VECTOR_LENGTH - 1) next_state = S_RELU;
                else next_state = S_FETCH;
            end
            S_RELU: next_state = S_OUTPUT;
            S_OUTPUT: begin
                if (internal_neuron_count == NUM_NEURONS - 1) next_state = S_DONE;
                else next_state = S_FETCH;
            end
            S_DONE: if (!start) next_state = S_IDLE;
            default: next_state = S_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_en <= 1'b0; mac_en <= 1'b0; acc_clr <= 1'b1; agu_inc <= 1'b0;
            agu_clr <= 1'b1; relu_en <= 1'b0; output_valid <= 1'b0; done <= 1'b0;
            internal_neuron_count <= 0;
        end else begin
            mem_en <= 1'b0; mac_en <= 1'b0; acc_clr <= 1'b0; agu_inc <= 1'b0;
            agu_clr <= 1'b0; relu_en <= 1'b0; output_valid <= 1'b0; done <= 1'b0;

            case (next_state)
                S_IDLE: begin
                    acc_clr <= 1'b1; agu_clr <= 1'b1;
                    internal_neuron_count <= 0;
                end
                S_FETCH: begin mem_en <= 1'b1; end
                S_MAC: begin mac_en <= 1'b1; agu_inc <= 1'b1; end
                S_RELU: begin relu_en <= 1'b1; end
                S_OUTPUT: begin
                    output_valid <= 1'b1;
                    acc_clr <= 1'b1;
                end
                S_DONE: begin done <= 1'b1; end
            endcase

            // Increment neuron count precisely when leaving S_OUTPUT for the next neuron
            if (current_state == S_OUTPUT && internal_neuron_count != NUM_NEURONS - 1) begin
                internal_neuron_count <= internal_neuron_count + 1'b1;
            end
        end
    end
endmodule
