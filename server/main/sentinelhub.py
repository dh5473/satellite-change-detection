from typing import List
import os
from datetime import datetime
from sentinelhub import (CRS,
                         BBox,
                         SHConfig,
                         MimeType,
                         DataCollection,
                         SentinelHubRequest,
                         SentinelHubDownloadClient,
                         SentinelHubCatalog,
                         bbox_to_dimensions)


def send_sentinel_request(
        coordinates: List,
        start_date="2023-01-01",
        end_date="2023-01-20",
        resolution=10,
        download_path="static/prepare/A"
):
    # Sentinel Hub configuration
    config = SHConfig()
    config.sh_client_id = ''  # Sentinel Hub account client ID
    config.sh_client_secret = ''  # Sentinel Hub account client secret
    config.instance_id = ''  # Sentinel Hub account instance ID

    resolution = resolution
    # [123.08, 0.05, 123.28, 0.25]
    bbox_coords = BBox(coordinates, CRS.WGS84)  # 좌측 경도, 아래쪽 위도, 우측 경도, 위쪽 위도
    bbox_size = bbox_to_dimensions(bbox_coords, resolution=resolution)

    time_interval = (start_date, end_date)

    request = SentinelHubRequest(
        data_folder=download_path,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
        bbox=bbox_coords,
        size=bbox_size,
        config=config
    )

    request.save_data(redownload=True)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")  # 예: '20231224_153045'
    new_filename = f"{formatted_time}"

    files = os.listdir(download_path)
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(download_path, x)))
    os.rename(os.path.join(download_path, latest_file), os.path.join(download_path, new_filename))

    return os.path.join(download_path, new_filename)


evalscript = """
//VERSION=3

function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04"]
    }],
    output: {
      bands: 3
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
"""