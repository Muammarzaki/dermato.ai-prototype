package data

import (
	"context"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type Chronic struct {
	ID        uuid.UUID `gorm:"type:uuid;primary_key;default:gen_random_uuid()" json:"id"`
	Body      string    `gorm:"type:json" json:"body"`
	Status    string    `gorm:"type:varchar(10);check:status IN ('success','fail')" json:"status"`
	CreatedAt time.Time `gorm:"type:timestamp;not null" json:"created_at"`
}

type ChronicRepository struct {
	db *gorm.DB
}

func NewChronicRepository(db *gorm.DB) *ChronicRepository {
	return &ChronicRepository{
		db: db,
	}
}

func (r *ChronicRepository) Create(ctx context.Context, chronic *Chronic) error {
	return r.db.WithContext(ctx).Create(chronic).Error
}

func (r *ChronicRepository) FindById(ctx context.Context, id string) (*Chronic, error) {
	var chronic Chronic
	err := r.db.WithContext(ctx).First(&chronic, "id = ?", id).Error
	if err != nil {
		return nil, err
	}
	return &chronic, nil
}

type Pagination struct {
	Page     int `json:"page"`
	PageSize int `json:"page_size"`
}

func (r *ChronicRepository) FindAll(ctx context.Context, pagination Pagination) ([]Chronic, error) {
	var chronics []Chronic
	offset := (pagination.Page - 1) * pagination.PageSize
	err := r.db.WithContext(ctx).Offset(offset).Limit(pagination.PageSize).Find(&chronics).Error
	if err != nil {
		return nil, err
	}
	return chronics, nil
}

func (r *ChronicRepository) Delete(ctx context.Context, id string) error {
	return r.db.WithContext(ctx).Delete(&Chronic{}, "id = ?", id).Error
}
