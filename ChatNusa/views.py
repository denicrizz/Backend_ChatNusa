from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .bot import get_response

@csrf_exempt
def search_api(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method harus POST"}, status=405)
    
    import json
    data = json.loads(request.body.decode("utf-8"))
    query = data.get("message", "")
    if not query:
        return JsonResponse({"status": "error", "message": "Query kosong."})

    result_type, result = get_response(query)

    if result_type == "info":
        return JsonResponse({
            "status": "success",
            "type": "info_UNP",
            "jawaban": result["jawaban"]
        })
    elif result_type == "repository":
        return JsonResponse({
            "status": "success",
            "type": "repository",
            "results": result
        })
    else:
        return JsonResponse({
            "status": "no_result",
            "message": "Maaf, tidak ditemukan informasi yang relevan."
        })
