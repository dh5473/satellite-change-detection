from django.shortcuts import render, redirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .sentinelhub import send_sentinel_request
from .apps import MainConfig
import json
import os


class MainView(APIView):
    def get(self, request):
        return render(request, 'main/main.html')

    def post(self, request):
        image_path = request.POST.get('selected_image')
        context = {
            'image_path': image_path,
        }
        return render(request, 'main/demo.html', context)


class DemoTest(APIView):
    def get(self, request):
        return render(request, 'main/prepare_1.html')

    def post(self, request):
        coordinates = request.POST.get('coordinates')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        try:
            coordinates = json.loads(coordinates)
        except json.JSONDecodeError:
            return Response({"error": "Invalid coordinates format"}, status=status.HTTP_400_BAD_REQUEST)

        if not coordinates or not isinstance(coordinates, list):
            return Response({"error": "Invalid or missing coordinates"}, status=status.HTTP_400_BAD_REQUEST)

        file_cnt_a = len(os.listdir("static/prepare/A"))
        file_cnt_b = len(os.listdir("static/prepare/B"))

        if file_cnt_a == file_cnt_b:
            try:
                first_path = send_sentinel_request(
                    coordinates=coordinates,
                    start_date=start_date,
                    end_date=end_date,
                    download_path="static/prepare/A"
                )

                return render(request, 'main/prepare_2.html')

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            try:
                second_path = send_sentinel_request(
                    coordinates=coordinates,
                    start_date=start_date,
                    end_date=end_date,
                    download_path="static/prepare/B"
                )

                files_a = os.listdir("static/prepare/A")
                files_b = os.listdir("static/prepare/B")
                first = f"static/prepare/A/{files_a[-1]}/response.png"
                second = f"static/prepare/B/{files_b[-1]}/response.png"

                MainConfig.model.inference(first, second)

                first_static_path = f"prepare/A/{files_a[-1]}/response.png"
                second_static_path = f"prepare/B/{files_b[-1]}/response.png"

                files_result = os.listdir("static/result")
                result_static_path = f"result/{files_result[-1]}/result.png"

                images = [first_static_path, second_static_path, result_static_path]

                context = {'images': images}
                return render(request, 'main/result.html', context)

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

