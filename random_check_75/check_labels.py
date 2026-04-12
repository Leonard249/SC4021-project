import json
import os

def check_discrepancies(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize counters
    post_stats = {'total': 0, 'checked': 0, 'accepted': 0, 'rejected': 0, 'label_changed': 0}
    comment_stats = {'total': 0, 'checked': 0, 'accepted': 0, 'rejected': 0, 'label_changed': 0}
    
    for post in data:
        post_stats['total'] += 1
        
        # Check post
        if 'user_check' in post and post['user_check']:
            post_stats['checked'] += 1
            uc = post['user_check']
            if uc.get('decision') == 'accept':
                post_stats['accepted'] += 1
            elif uc.get('decision') == 'reject':
                post_stats['rejected'] += 1
            
            # Count how many actually had a different final_label vs original Overall_Document_Polarity
            if uc.get('final_label') != post.get('Overall_Document_Polarity'):
                post_stats['label_changed'] += 1

        # Check comments
        for comment in post.get('Comments', []):
            comment_stats['total'] += 1
            if 'user_check' in comment and comment['user_check']:
                comment_stats['checked'] += 1
                uc = comment['user_check']
                if uc.get('decision') == 'accept':
                    comment_stats['accepted'] += 1
                elif uc.get('decision') == 'reject':
                    comment_stats['rejected'] += 1
                    
                if uc.get('final_label') != comment.get('Overall_Document_Polarity'):
                    comment_stats['label_changed'] += 1
                    
    print(f"--- POST STATISTICS ---")
    print(f"Total Posts    : {post_stats['total']}")
    print(f"User Checked   : {post_stats['checked']}")
    print(f"  -> Accepted  : {post_stats['accepted']}")
    print(f"  -> Rejected  : {post_stats['rejected']}")
    print(f"Label Changed  : {post_stats['label_changed']} (Final label differed from Overall_Document_Polarity)")
    print()
    print(f"--- COMMENT STATISTICS ---")
    print(f"Total Comments : {comment_stats['total']}")
    print(f"User Checked   : {comment_stats['checked']}")
    print(f"  -> Accepted  : {comment_stats['accepted']}")
    print(f"  -> Rejected  : {comment_stats['rejected']}")
    print(f"Label Changed  : {comment_stats['label_changed']} (Final label differed from Overall_Document_Polarity)")

if __name__ == '__main__':
    # Script assumes it is running in the 'random_check_75' directory alongside 'sample75.json'
    file_path = 'sample75.json'
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), 'sample75.json')
    
    print(f"Reading from {file_path}...")
    check_discrepancies(file_path)
