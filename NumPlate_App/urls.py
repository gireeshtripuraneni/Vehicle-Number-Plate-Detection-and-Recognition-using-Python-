from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .import views

urlpatterns = [
				path('',views.Home,name="Home"),
				path('Detect_Image/',views.Detect_Image,name="Detect_Image"),
				path('Detect_Video/',views.Detect_Video,name="Detect_Video"),
				path('Real_Time/',views.Real_Time,name="Real_Time"),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)