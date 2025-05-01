from django.http import JsonResponse
from .bot import get_response

def search_api(request):
    query = request.GET.get("message", "")
    if not query:
        return JsonResponse({"status": "error", "message": "Query kosong."})

    result_type, result = get_response(query)

    if result_type == "info":
        return JsonResponse({
            "status": "success",
            "type": "info",
            "pertanyaan": result["pertanyaan"],
            "jawaban": result["jawaban"]
        })
    elif result_type == "repository":
        return JsonResponse({
            "status": "success",
            "type": "repository",
            "results": result  # <- result adalah list of dicts
        })
    else:
        return JsonResponse({
            "status": "no_result",
            "message": "Maaf, tidak ditemukan informasi yang relevan."
        })
