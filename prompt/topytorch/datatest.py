
from spec import DataSpec


dataspec = DataSpec('crello', '/home/usr/dell/Project-HCL/BaseLine/flex-dm/data/crello', batch_size=16)

# dataspec = DataSpec('dataforfd', '/storage/homes/hzj/hzj/Project-HCL/BaseLine/flex-dm/data/DataForFlexDm/tfrecords_output_v5', batch_size=16)


train_dataset = dataspec.make_dataset(
    'train', shuffle=True, cache=True)


for _ in range(10):
    batch = next(iter(train_dataset))

        # Print sample
    print("\nSample data:")
    for item in dataspec.unbatch(batch):
        print(f"Sample ID: {item['id']}")
        print(f"Canvas: {item['canvas_width']}x{item['canvas_height']}")
        print(f"Elements: {len(item['elements'])}")
        for i, elem in enumerate(item['elements'][:3]):  # Show first 3 elements
            print(f"  Element {i}: {elem['type']} at ({elem['left']:.3f}, {elem['top']:.3f})")
            print(f"    Size: {elem['width']:.3f} x {elem['height']:.3f}")
        if len(item['elements']) > 3:
            print(f"  ... and {len(item['elements']) - 3} more elements")
        break



# for item in dataspec.unbatch(batch):
#     print(item)