#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
struct NibbleReader {
  const std::uint8_t* data;
  std::size_t size;
  std::size_t offset = 0;
  std::uint32_t word = 0;
  int nibbles_left = 0;

  bool next_vle(std::uint64_t& value) {
    value = 0;
    int shift = 0;
    while (true) {
      if (nibbles_left == 0) {
        if (offset >= size) return false;
        word = 0;
        for (int byte = 0; byte < 4 && offset + byte < size; ++byte) {
          word |= static_cast<std::uint32_t>(data[offset + byte]) << (8 * byte);
        }
        offset += 4;
        nibbles_left = 8;
      }
      const std::uint32_t nibble = (word >> 28) & 0x0fU;
      word <<= 4;
      --nibbles_left;
      if (shift > 63 || (shift == 63 && (nibble & 0x06U) != 0)) return false;
      value |= static_cast<std::uint64_t>(nibble & 0x07U) << shift;
      if ((nibble & 0x08U) == 0) return true;
      shift += 3;
    }
  }
};
}  // namespace

extern "C" int ego_loong_decode_rvl_u16(const std::uint8_t* payload,
                                         std::size_t payload_size,
                                         std::uint16_t* output,
                                         std::size_t pixel_count) {
  if (payload == nullptr || output == nullptr) return -1;
  NibbleReader reader{payload, payload_size};
  std::size_t pixel = 0;
  std::int64_t previous = 0;
  while (pixel < pixel_count) {
    std::uint64_t zeros = 0;
    if (!reader.next_vle(zeros)) return -2;
    if (zeros > pixel_count - pixel) return -3;
    for (std::uint64_t i = 0; i < zeros; ++i) output[pixel++] = 0;

    std::uint64_t nonzeros = 0;
    if (!reader.next_vle(nonzeros)) return -2;
    if (nonzeros > pixel_count - pixel) return -4;
    for (std::uint64_t i = 0; i < nonzeros; ++i) {
      std::uint64_t positive = 0;
      if (!reader.next_vle(positive)) return -2;
      const std::int64_t delta = static_cast<std::int64_t>(positive >> 1) ^
                                 -static_cast<std::int64_t>(positive & 1U);
      previous += delta;
      if (previous < 0 || previous > std::numeric_limits<std::uint16_t>::max()) return -5;
      output[pixel++] = static_cast<std::uint16_t>(previous);
    }
  }
  return 0;
}
