import pandas as pd
import re
from googleapiclient.discovery import build

import warnings # 경고창 무시
warnings.filterwarnings('ignore')

### 카테고리 별로 총 20000개 crawling
VIDEO_ID = "68C679HqGbs"

comments = list()
## 고유 YouTube data API v3
api_obj = build('youtube', 'v3', developerKey='AIzaSyAbDh3SBjRvpUzDbGNLyQo5jQaemiQGClA')

# videoid -> 유튜브 고유 아이디
# maxResults: 한 번의 요청에서 최대 100개 댓글을 가져옴
# execute: API 요청 실행 및 JSON 형식의 응답 반환
response = api_obj.commentThreads().list(part='snippet,replies', videoId=VIDEO_ID, maxResults = 100).execute()

while response:
    for item in response['items']:
        # comment 부분만 추출
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    # 다음 페이지 확인 후 가져오기
    if 'nextPageToken' in response:
        response = api_obj.commentThreads().list(
            part = 'snippet',  # 'replies' 제거
            videoId = VIDEO_ID,
            pageToken = response['nextPageToken'],
            maxResults = 100
        ).execute()
    else:
        break
    
for i, comment in enumerate(comments[:10]):  # 10개만 출력
    print(f"{i+1}. {comment}")
    
df = pd.DataFrame(comments)
df.to_excel('results.xlsx', header=['comment'], index=None)


