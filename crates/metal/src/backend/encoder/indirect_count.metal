#include <metal_stdlib>
using namespace metal;

struct Commands { command_buffer buffer [[id(0)]]; };
struct Parameters {
    ulong argument_word_offset;
    ulong count_word_offset;
    ulong index_byte_offset;
    ulong index_elements;
    uint stride_words;
    uint first_command;
    uint maximum;
    uint chunk_size;
    uint index_size;
};

kernel void fluxel_generate_draws(
    constant Commands &commands [[buffer(0)]],
    device const uint *arguments [[buffer(1)]],
    device const uint *count [[buffer(2)]],
    device uchar *indices [[buffer(3)]],
    constant Parameters &params [[buffer(4)]],
    uint local_id [[thread_position_in_grid]]) {
    uint id = params.first_command + local_id;
    if (local_id >= params.chunk_size || id >= min(params.maximum, count[params.count_word_offset]))
        return;
    device const uint *draw = arguments + params.argument_word_offset + ulong(id) * params.stride_words;
    render_command command(commands.buffer, local_id);
    if (params.index_size == 0) {
        command.draw_primitives(primitive_type::triangle, draw[2], draw[0], draw[1], draw[3]);
    } else {
        ulong first = draw[2];
        if (first > params.index_elements || ulong(draw[0]) > params.index_elements - first)
            return;
        device uchar *base = indices + params.index_byte_offset + first * params.index_size;
        // MSL takes base_vertex as uint; preserve the signed indirect word's bit pattern.
        if (params.index_size == 2) {
            command.draw_indexed_primitives(primitive_type::triangle, draw[0],
                (device ushort *)base, draw[1], draw[3], draw[4]);
        } else {
            command.draw_indexed_primitives(primitive_type::triangle, draw[0],
                (device uint *)base, draw[1], draw[3], draw[4]);
        }
    }
}
